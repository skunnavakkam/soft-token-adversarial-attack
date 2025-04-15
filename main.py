import argparse
from transformer_lens import HookedTransformer
import torch
from typing import Tuple, List
import torch.nn.functional as F
import math


def exec_model(model: HookedTransformer, first_tok_embedding, toks):
    res, _tks, _spe, _attn = model.input_to_embed(toks)
    total = torch.concat([first_tok_embedding, res], axis=1)
    return model(total, start_at_layer=0)


def predict(
    model: HookedTransformer, first_tok_embedding, num_toks: int
) -> Tuple[str, List[str], torch.Tensor]:
    tokens = []
    for i in range(num_toks):
        toks = torch.tensor(tokens, dtype=torch.long)
        res, _tks, _spe, _attn = model.input_to_embed(toks)
        total = torch.concat([first_tok_embedding, res], axis=1)

        res = model(total, start_at_layer=0)[0]
        next_token = torch.argmax(res, axis=-1)[-1].item()

        tokens.append(next_token)

    return (
        model.tokenizer.decode(tokens),
        model.tokenizer.convert_ids_to_tokens(tokens),
        torch.tensor(tokens).cpu(),
    )


def loss_fn(logits, tokens, first_tok_embedding, gamma=0.2):
    def l2(x):
        return torch.sum(x**2) ** 0.5

    logits = logits[0:-1, :]

    ce_loss = F.cross_entropy(logits, tokens)
    l2_loss = l2(logits)

    total = ce_loss + gamma * l2_loss

    return total


def attack(text: str, model: str = "gpt2", soft_tokens=1, num_steps=1000):
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = HookedTransformer.from_pretrained(model, device=device)

    toks = model.to_tokens(text)[:, 1:]
    model_dim = model.cfg.d_model
    soft_token_embeddings = torch.randn(
        size=(1, soft_tokens, model_dim), dtype=torch.float32, device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    last_corr = 0
    max_corr = 0
    lookahead = 5

    for step in range(num_steps):
        optimizer.zero_grad()

        logits = exec_model(model, soft_token_embeddings, toks)
        flattened_logits = logits.flatten(0, 1)[: last_corr + lookahead + soft_tokens]
        flattened_tokens = toks.flatten(0, 1)[: last_corr + lookahead]

        loss = loss_fn(flattened_logits, flattened_tokens, soft_token_embeddings)

        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            with torch.no_grad():
                # now we run a prediction!
                ps = predict(model, soft_token_embeddings, last_corr + lookahead)

                temp_corr = (
                    (
                        ps[2][: last_corr + lookahead]
                        == (toks[:, : last_corr + lookahead]).cpu()
                    )
                    .cpu()
                    .sum()
                )

                max_corr = max(max_corr, temp_corr)

                if temp_corr > last_corr:
                    last_corr += math.ceil((temp_corr - last_corr) / 2)

                print(
                    f"Step {step} | Corr: {temp_corr} | Max Corr: {max_corr} | Loss: {loss.item()} | {ps[0]}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    attack(args.text)
