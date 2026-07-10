import os

import datasets


_CITATION = """\\
@InProceedings{wikitext,
  author={Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
  year={2016}
}
"""

_DESCRIPTION = """\\
WikiText is a word-level language modeling dataset extracted from verified
Wikipedia articles.
"""

_HOMEPAGE = "https://huggingface.co/datasets/wikitext"

_LICENSE = "CC BY-SA 3.0"

_RAW_V1_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip?download=true"


def _resolve_local_archive():
    candidates = [
        os.path.join(os.getcwd(), "datasets", "wikitext", "wikitext-2-raw-v1.zip"),
        os.path.join(os.path.dirname(__file__), "wikitext-2-raw-v1.zip"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class Wikitext(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="wikitext-2-raw-v1",
            version=VERSION,
            description="WikiText-2 raw dataset.",
        )
    ]
    DEFAULT_CONFIG_NAME = "wikitext-2-raw-v1"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"text": datasets.Value("string")}),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        local_archive = _resolve_local_archive()
        archive_path = local_archive or dl_manager.download(_RAW_V1_URL)
        data_dir = dl_manager.extract(archive_path)
        base_dir = os.path.join(data_dir, "wikitext-2-raw")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(base_dir, "wiki.train.raw")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(base_dir, "wiki.valid.raw")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(base_dir, "wiki.test.raw")},
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                yield idx, {"text": line.rstrip("\n")}
