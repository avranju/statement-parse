import argparse
import io
import base64
import json
from pdf2image import convert_from_path
from PIL import Image
import requests


def main():
    parser = make_args_parser()
    args = parser.parse_args()
    if not args.pdf_path or not args.ollama_url or not args.model:
        parser.print_help()
        return

    images = convert_from_path(args.pdf_path, size=(1092, None))
    image_strs = list(map(image_to_base64, range(0, len(images)), images))
    prompt = make_prompt(image_strs, args.model)

    url = args.ollama_url
    with requests.post(url, json=prompt, stream=True) as response:
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return
        buffer = ""
        done = False
        for chunk in response.iter_content(chunk_size=1024):
            buffer += chunk.decode("utf-8")
            lines = buffer.split("\n")
            for line in lines[:-1]:
                response, done = process_line(line)
                print(response, end="")
            if done:
                break
            buffer = lines[-1]
        if not done and buffer:
            response, _ = process_line(buffer)
            print(response, end="")


def process_line(line: str):
    res = json.loads(line)
    response = res["response"]
    done = res["done"]
    return response, done


def image_to_base64(i, image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image.save(f"output_{i}.png")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_prompt(image_strs: list[str], model_name: str) -> dict:
    return {
        "model": model_name,
        "prompt": """I would like you to analyze the text in the attached image(s).
                     I would like you to generate a listing of share deposits along with
                     date and price. Consider dividend reinvestments as share deposits.
                     Return the list of deposits as a JSON document.""",
        "images": image_strs,
    }


def make_args_parser():
    parser = argparse.ArgumentParser(
        description="Extract deposit information from investment account statement PDFs"
    )
    parser.add_argument("-p", "--pdf-path", type=str, help="Path to PDF file")
    parser.add_argument("-u", "--ollama-url", type=str, help="URL to Ollama API")
    parser.add_argument("-m", "--model", type=str, default="llava", help="Model to use")
    return parser


if __name__ == "__main__":
    main()
