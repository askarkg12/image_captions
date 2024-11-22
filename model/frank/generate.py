import torch
import torchvision
import torch.nn.functional as F
from PIL.Image import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


class CaptionGenerator:
    def __init__(
        self,
        *,
        model,
        tokeniser,
        default_max_length: int = 50,
        default_temp: float = 1.0,
        default_rep_penalty: float = 1.0,
        default_top_k: int = 50,
    ) -> None:
        self.model = model
        self.img_to_tensor = (
            torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        )
        self.tokeniser = tokeniser
        self.default_max_length = default_max_length
        self.default_temp = default_temp
        self.default_rep_penalty = default_rep_penalty
        self.default_top_k = default_top_k

    def generate(
        self,
        image: torch.Tensor | Image,
        *,
        max_length=None,
        temperature=None,
        repetition_penalty=None,
        top_k=None,
    ):
        if isinstance(image, Image):
            image = self.img_to_tensor(image).unsqueeze(0).to(device)
        if max_length is None:
            max_length = self.default_max_length
        if temperature is None:
            temperature = self.default_temp
        if repetition_penalty is None:
            repetition_penalty = self.default_rep_penalty
        if top_k is None:
            top_k = self.default_top_k

        # Start with BOS token
        current_tokens = torch.tensor([[self.tokeniser.bos_id]], device=device)

        # Generate tokens until EOS or max_length
        while current_tokens.size(1) < max_length:
            # Get model predictions for next token
            with torch.inference_mode():
                outputs = self.model(image, current_tokens)
                next_token_logits = outputs[:, -1, :]

            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature

            # TODO Apply repetition penalty
            if repetition_penalty != 1.0:
                for token in set(current_tokens[0].tolist()):
                    next_token_logits[0, token] /= repetition_penalty

            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)

            # Convert logits to probabilities
            probs = F.softmax(top_k_logits, dim=-1)

            # Sample next token
            next_token = top_k_indices[0, torch.multinomial(probs, num_samples=1)]

            # Append next token to sequence
            current_tokens = torch.cat([current_tokens, next_token], dim=1)

            # Check if EOS token was generated
            if next_token.item() == self.tokeniser.eos_id:
                break

        tkn_seq = current_tokens[0].tolist()
        text = self.tokeniser.decode(tkn_seq)

        return tkn_seq, text
