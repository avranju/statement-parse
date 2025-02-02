import argparse
import io
import fnmatch
import os
import base64
import json
import sys
from pdf2image import convert_from_path
from PIL import Image
from functools import partial
import anthropic


def main():
    parser = make_args_parser()
    args = parser.parse_args()
    if not args.pdf_path or not args.api_key:
        parser.print_help()
        return

    pdf_files = list_pdf_files(args.pdf_path, args.file_names)
    to_base64 = partial(image_to_base64, dump_files=args.dump_images)

    for pdf_file in pdf_files:
        log(f"Processing {pdf_file}")
        images = convert_from_path(pdf_file, size=(1092, None))
        image_strs = list(map(to_base64, range(0, len(images)), images))
        prompt = make_prompt(image_strs)

        response = anthropic.Anthropic(api_key=args.api_key).messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=2048, messages=prompt
        )
        text_output = list(
            map(lambda x: x.text, filter(lambda x: x.type == "text", response.content))
        )
        log(" ".join(text_output))
        data = json.loads(extract_json_content(" ".join(text_output)))
        print("\n".join([f'"{f["date"]}",{f["shares"]},{f["price"]}' for f in data]))
        sys.stdout.flush()


def process_line(line: str):
    res = json.loads(line)
    response = res["response"]
    done = res["done"]
    return response, done


def image_to_base64(i, image: Image.Image, dump_files) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    if dump_files:
        image.save(f"output_{i}.png")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_image_message(image: str) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": image,
        },
    }


def make_prompt(image_strs: list[str]):
    prompt = [
        {
            "role": "user",
            "content": """I would like you to analyze the text in the attached image(s).
                        I would like you to generate a listing of share deposits along with
                        date and price. Consider dividend reinvestments as share deposits.
                        Return the list of deposits as a JSON document. Emit the string BEGIN_JSON
                        on a separate line before the JSON output starts and emit END_JSON on a
                        separate line after the JSON output ends. The JSON output should be
                        a JSON array of objects, each with a 'date', 'shares' and 'price' field.
                        The 'date' field should be a string in the format 'YYYY-MM-DD'. The 'shares'
                        and 'price' fields should be numbers. For example:
                            [{"date": "2022-01-01", "shares": 19.000, "price": 43.8500}, ...]""",
        },
        {
            "role": "user",
            "content": map(make_image_message, image_strs),
        },
    ]

    return prompt


def make_args_parser():
    parser = argparse.ArgumentParser(
        description="Extract deposit information from investment account statement PDFs using Claude AI"
    )
    parser.add_argument(
        "-p", "--pdf-path", type=str, help="Path to folder containing PDF file"
    )
    parser.add_argument(
        "-f",
        "--file-names",
        type=str,
        nargs="*",
        help="PDF file names to process; if not provided, all PDF files in the folder will be processed",
    )
    parser.add_argument("-k", "--api-key", type=str, help="Claude API key")
    parser.add_argument(
        "-d", "--dump-images", help="Dump images to disk", action="store_true"
    )
    return parser


def list_pdf_files(dir, file_names):
    pdf_files = []
    if file_names:
        pdf_files += file_names
    else:
        pdf_files = [f for f in os.listdir(dir) if fnmatch.fnmatch(f, "*.pdf")]
    return [os.path.join(dir, f) for f in pdf_files]


def extract_json_content(text):
    # Split the text into lines
    lines = text.splitlines()

    # Initialize variables
    json_started = False
    extracted_lines = []

    for line in lines:
        if "BEGIN_JSON" in line:
            json_started = True
            continue  # Skip the BEGIN_JSON marker itself

        if "END_JSON" in line:
            break  # Stop when END_JSON is encountered

        if json_started:
            extracted_lines.append(line)

    return "\n".join(extracted_lines)


def log(s):
    print(s, file=sys.stderr)


if __name__ == "__main__":
    main()
