import jittor as jt
from jittor import nn

class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = jt.zeros(prompt_pool_shape)
            elif prompt_init == 'uniform':
                self.prompt = jt.init.uniform(prompt_pool_shape, 'float32', -1, 1)
            self.prompt = jt.Var(self.prompt)  # Wrap as trainable parameter
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = jt.zeros(key_shape)
            elif prompt_key_init == 'uniform':
                self.prompt_key = jt.init.uniform(key_shape, 'float32', -1, 1)
            self.prompt_key = jt.Var(self.prompt_key)  # Wrap as trainable parameter
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = jt.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = jt.sum(x ** 2, dim=dim, keepdims=1)
        x_inv_norm = jt.rsqrt(jt.maximum(square_sum, jt.array(epsilon)))
        return x * x_inv_norm
    
    def execute(self, x_embed, prompt_mask=None, cls_features=None):
        out = {}
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = jt.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = jt.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = jt.max(x_embed, dim=1)[0] + 2 * jt.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = jt.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = x_embed_norm @ prompt_norm.t() # B, Pool_size
            
            if prompt_mask is None:
                _, idx = jt.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = jt.unique(idx.flatten(), return_counts=True, sorted=True)
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = jt.concat([prompt_id, jt.full((self.pool_size - prompt_id.shape[0],), jt.min(idx.flatten()))])
                        id_counts = jt.concat([id_counts, jt.full((self.pool_size - id_counts.shape[0],), 0)])
                    _, major_idx = jt.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand((x_embed.shape[0], -1)) # B, top_k
            else:
                idx = prompt_mask # B, top_k

            batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = jt.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = jt.zeros((self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = jt.init.uniform((self.length, self.embed_dim), 'float32', -1, 1)
            self.prompt = jt.Var(self.prompt)  # Wrap as trainable parameter
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = jt.concat([batched_prompt, x_embed], dim=1)

        return out
