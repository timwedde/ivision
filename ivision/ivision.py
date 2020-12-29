### System ###
import io
import re
import sys
import logging
from pathlib import Path

### Logging ###
from rich import print
from rich.traceback import install
from rich.logging import RichHandler

install()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)

logger = logging.getLogger("rich")

### Apple Frameworks ###
import Quartz
import Vision

### Data Handling ###
import pandas as pd

### CLI Parsing ###
import click


def pil_to_cimg(image, alpha=False):
    """Load an image given a PIL Image object.
    This has some conversion overhead because the file needs to be
    re-encoded to JPG or PNG before loading it.
    """
    saved_image = io.BytesIO()
    image.save(saved_image, format="png" if alpha else "jpeg")
    saved_image = saved_image.getbuffer()
    provider = Quartz.CGDataProviderCreateWithData(
        None, saved_image, len(saved_image), None
    )
    return Quartz.CGImageCreateWithJPEGDataProvider(provider, None, None, 0)


def file_to_cimg(file):
    """Load an image given a file path.
    This is much faster than the PIL conversion because it does not have
    the conversion overhead.
    """
    url = Quartz.NSURL.alloc().initFileURLWithPath_(file)
    provider = Quartz.CGDataProviderCreateWithURL(url)
    return Quartz.CGImageCreateWithJPEGDataProvider(provider, None, None, 0)


def ocr_image(image, languages=None, language_correction=True, fast=False):
    """OCR's an image."""
    if isinstance(image, str):
        cg_img = file_to_cimg(image)
    else:
        cg_img = pil_to_cimg(image)

    data_text, data_words = None, None
    ocr_data_text, ocr_data_words = [], []

    def output_handler(req, err):
        for result in req.results():
            cand = result.topCandidates_(1)[0]
            string = cand.string()
            confidence = cand.confidence()

            rect, err = cand.boundingBoxForRange_error_(
                Quartz.NSRange(0, len(string)), None
            )
            box = rect.boundingBox()

            box = Vision.VNImageRectForNormalizedRect(box, width, height)
            x, y = box.origin.x, box.origin.y
            w, h = box.size.width, box.size.height
            ocr_data_text.append((x, height - y - h, w, h, confidence, string))

            for match in re.finditer(r"(\S+)", string):
                start, end = match.span()
                rect, err = cand.boundingBoxForRange_error_(
                    Quartz.NSRange(start, end - start), None
                )
                box = rect.boundingBox()

                box = Vision.VNImageRectForNormalizedRect(box, width, height)
                x, y = box.origin.x, box.origin.y
                w, h = box.size.width, box.size.height

                ocr_data_words.append(
                    (x, height - y - h, w, h, confidence, match.group())
                )

    request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(
        output_handler
    )
    request.setRecognitionLanguages_(languages or ["en"])
    request.setUsesLanguageCorrection_(language_correction)
    request.setRecognitionLevel_(1 if fast else 0)

    width = Quartz.CGImageGetWidth(cg_img)
    height = Quartz.CGImageGetHeight(cg_img)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cg_img, None
    )
    handler.performRequests_error_([request], None)

    if ocr_data_text:
        data_text = pd.DataFrame.from_records(ocr_data_text)
        data_text.columns = ["left", "top", "width", "height", "confidence", "text"]
        data_text["confidence"] = data_text["confidence"].round(2)

    if ocr_data_words:
        data_words = pd.DataFrame.from_records(ocr_data_words)
        data_words.columns = ["left", "top", "width", "height", "confidence", "text"]
        data_words["confidence"] = data_words["confidence"].round(2)

    return data_text, data_words


### CLI ###


class AliasedGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        matches = [x for x in self.list_commands(ctx) if x.startswith(cmd_name)]
        if not matches:
            return None
        elif len(matches) == 1:
            return click.Group.get_command(self, ctx, matches[0])
        ctx.fail("Too many matches: {}".format(", ".join(sorted(matches))))


@click.group(cls=AliasedGroup)
@click.option("-d", "--debug", is_flag=True)
def cli(debug):
    """Commands can be abbreviated by the shortest unique string.

    \b
    For example:
        ocr -> o
    """
    logger.setLevel(logging.DEBUG if debug else logging.INFO)


@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.argument(
    "file", required=True, type=click.Path(dir_okay=False, resolve_path=True)
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, resolve_path=True, writable=True),
    default=None,
    help="The output to save to",
)
@click.option("-f", "--fast", is_flag=True, help="Use fast OCR mode")
@click.option("-w", "--words", is_flag=True, help="Output bounding boxes on word-level")
@click.option("-n", "--no-correct", is_flag=True, help="Do not use language correction")
@click.option(
    "-l",
    "--lang",
    "languages",
    type=str,
    multiple=True,
    help="Language of the document (can be multiple)",
)
def ocr(file, output, fast, words, no_correct, languages):
    """OCR the given image file."""
    data_text, data_words = ocr_image(
        file, fast=fast, languages=languages, language_correction=not no_correct
    )

    output_data = data_words if words else data_text

    if not output:
        print("\n".join(output_data["text"].values))
        return

    output = Path(output)
    if output.suffix == ".txt":
        with output.open("w") as f:
            f.write("\n".join(output_data["text"].values))
    elif output.suffix == ".csv":
        output_data.to_csv(output)
    else:
        print("No file suffix, could not guess file format")
        sys.exit(1)


if __name__ == "__main__":
    cli()
