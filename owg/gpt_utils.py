import os
import numpy as np
import base64
import requests
from io import BytesIO
from typing import List, Union, Optional, Any
from PIL import Image
import json, re, ast, logging


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
logging.basicConfig(level=logging.INFO)  # 或 DEBUG


def parse_llm_payload(resp: Any) -> Optional[Any]:
    """
    Make LLM outputs robust:
    - if resp is already list/dict/int -> return directly
    - if resp is string -> extract codeblock / find JSON-ish substring
    - try json.loads, then ast.literal_eval as fallback (handles single quotes)
    """
    if resp is None:
        return None

    # Already structured
    if isinstance(resp, (dict, list, int, float, bool)):
        return resp

    # Force to string
    if not isinstance(resp, str):
        resp = str(resp)

    s = resp.strip()
    if not s:
        return None

    # If wrapped by ```json ... ```
    m = _JSON_BLOCK_RE.search(s)
    if m:
        s = m.group(1).strip()

    # 1) try pure json
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) try extract first {...} or [...]
    m2 = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", s)
    if m2:
        cand = m2.group(1).strip()
        try:
            return json.loads(cand)
        except Exception:
            pass
        try:
            return ast.literal_eval(cand)  # handles single quotes
        except Exception:
            pass

    # 3) last resort: literal_eval whole string
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _extract_json_from_text(text: str):
    """
    Try to find the first JSON object/array inside a text blob and parse it.
    Returns parsed object or None.
    """
    # 1) Try to find a ```json ... ``` or ``` ... ``` block
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 2) Try to find first {...} or [...] in the text (greedy edges)
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        candidate = m.group(1)
        # attempt balanced brace extraction: try shrinking end until valid json
        for end in range(len(candidate), 0, -1):
            try:
                return json.loads(candidate[:end])
            except Exception:
                continue

    # 3) no JSON found
    return None




# Get OpenAI API Key from environment variable
openai_api_key = os.environ.get('OPENAI_API_KEY', 'sk-A9q5TscQFLIV7ZTAB29f5c93E1D44f4880F91c24FcAa4eDd')

API_URL = "https://api.openai.com/v1/chat/completions"


def encode_image_to_base64(image) -> str:
    """
    Encodes an image into a base64-encoded string in JPEG format.

    Parameters:
        image (np.ndarray): The image to be encoded. This will be a string
        of the image path or a PIL image

    Returns:
        str: A base64-encoded string representing the image in JPEG format.
    """
    # Function to encode the image
    def _encode_image_from_file(image_path):
        # Function to encode the image
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    def _encode_image_from_pil(image):
        buffered = BytesIO()
        image.save(buffered, format='JPEG')
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    if isinstance(image, str):
        return _encode_image_from_file(image)
    elif isinstance(image, Image.Image):
        return _encode_image_from_pil(image)
    elif isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
        return _encode_image_from_pil(image_pil)
    else:
        raise ValueError(f"Unknown option for image {type(image)}")


def prepare_prompt(
    images: List[Union[Image.Image, np.ndarray]],
    prompt: Optional[str] = None,
    in_context_examples: Optional[dict] = None
  ) -> dict:
  
  def _append_pair(current_prompt, images, text):
    # text first if given, then image.
    if text:
      current_prompt['content'].append({
          'type': 'text',
          'text': text
        })
    else:
      assert len(images) > 0, "Both images and text prompts are empty."

    for image in images:
      base64_image = encode_image_to_base64(image)
      current_prompt['content'].append({
              "type": "image_url",
              "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image}",
                  #"detail": "low"
          }
        })
    return current_prompt

  set_prompt = {
    'role': 'user',
    'content': []
  }

  # Include in-context examples if provided
  if in_context_examples:
    for example in in_context_examples:
      _append_pair(
        set_prompt, example['images'], example['prompt'])
      # interleave response
      set_prompt['content'].append({
          'type': 'text',
          'text': f"The answer should be: {example['response']}\n"
      })
    
  # add user prompt
  _append_pair(set_prompt, images, prompt)

  return set_prompt


# def prepare_prompt(
#         images: List[np.ndarray], 
#         prompt: Optional[str] = None, 
#         detail: str = "auto"
# ) -> dict:
#    # text prompt always goes first, then images prompt
#     set_user_prompt = {
#         "role": "user",
#         "content": []
#     }

#     if not text_prompt:
#       assert len(images) > 0, "Image and text prompts are both empty."
#     else:
#       set_user_prompt["content"].append({
#         "type": "text",
#         "text": prompt
#       })

#     # If there are no images, return a simple text prompt
#     if not images:
#         return set_user_prompt
    
#     # Otherwise, prepare prompt with images    
#     for image in images:
#         base64_image = encode_image_to_base64(image)
#         image_prompt = {
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{base64_image}",
#                 "detail": detail
#             }
#         }
#         set_user_prompt["content"].append(image_prompt)
#     return set_user_prompt


def compose_payload(images: List[np.ndarray], prompt: str, system_prompt: str, detail: str, temperature: float, max_tokens: int, n: int, model_name: str = "gpt-4o", return_logprobs: bool = False, in_context_examples: List[dict] = None, seed: Optional[int] = None) -> dict:
    # Prepare system message
    system_msg = {
                "role": "system",
                "content": system_prompt  # plain text, not a list
    }
    messages = [system_msg]
    # Prepare prompt message, potentially with in-context examples
    msg = prepare_prompt(
      images, prompt, in_context_examples)
    messages.append(msg)
    
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n,
        "logprobs": return_logprobs,
    }
    # reproducable output?
    if seed is not None:
      payload["seed"] = seed
    return payload


def request_gpt(images: Union[np.ndarray, List[np.ndarray]], prompt: str, system_prompt: str, detail: str = "auto", temp: float = 0.0, n_tokens: int = 256, n: int = 1, return_logprobs: bool = False, in_context_examples: List[dict] = None, model_name: str = "gpt-4o", seed: Optional[int] = None) -> str:
    api_key = "sk-A9q5TscQFLIV7ZTAB29f5c93E1D44f4880F91c24FcAa4eDd"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # convert single image prompt to multiple for compatibility
    if not isinstance(images, List):
        assert isinstance(images, np.ndarray), "Provide either a numpy array, a PIL image, an image path string or a list of the above."
        images = [images]
    
    payload = compose_payload(
        images=images,
        prompt=prompt,
        detail=detail,
        system_prompt=system_prompt,
        n=n,
        temperature=temp,
        max_tokens=n_tokens,
        return_logprobs=return_logprobs,
        in_context_examples=in_context_examples,
        model_name=model_name,
        seed=seed
    )
    
    response = requests.post("https://api.ai-yyds.com/v1/chat/completions", json=payload, headers=headers)

    logging.info("LLM raw status: %s", response.status_code)
    logging.debug("LLM raw response text:\n%s", response.text)

    if response.status_code == 200:
        res_json = response.json()
        logging.info("LLM raw json: %s", res_json)

        content = ""
        try:
            choice0 = res_json["choices"][0]

            # chat/completions 常见格式
            msg = choice0.get("message", {})
            content = msg.get("content")

            # 有些代理接口会把文本放在 text
            if content is None:
                content = choice0.get("text", "")

            # 防止 content 仍然是 None
            if content is None:
                content = ""

        except Exception as e:
            logging.error("Failed to parse LLM response json: %s", e)
            content = ""

        parsed = None
        if isinstance(content, str) and content.strip():
            parsed = _extract_json_from_text(content)

        if parsed is not None:
            return parsed
        elif isinstance(content, str) and content.strip():
            return content
        else:
            logging.error("LLM returned empty content. Full json: %s", res_json)
            return ""
    else:
        logging.error("LLM request failed: %s", response.text)
        raise ValueError(f"请求失败，状态码：{response.status_code}，错误信息：{response.text}")

