from sq_datasets import ScoreboardDataset
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw


class CRAFTSBDataset:
    def __init__(self, images_dir) -> None:
        self.images_dir = images_dir
        self.annotations_dir = images_dir.replace(
            "images", "localization_transcription_gt"
        )

        self.list_images = list(Path(self.images_dir).glob("*.jpg"))

    def __getitem__(self, index):
        image_path = self.list_images[index]
        image = Image.open(image_path)
        annotation = self._get_annotation(index)

        return image_path, image, annotation

    def __len__(self):
        return len(self.list_images)

    def _get_annotation(self, index):
        image_name = self.list_images[index].name
        annotation_path = (
            Path(self.annotations_dir) / f"gt_{image_name.replace('.jpg', '.txt')}"
        )

        with open(annotation_path, "r") as f:
            lines = f.readlines()

            annotations = []
            for line in lines:
                line = line.strip().split(",")
                annotations.append(line)

        return annotations


def func_draw_boxes(image, box, width=2, color=(0, 255, 0)):
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, width=width, outline=color)
    return image


if __name__ == "__main__":
    dataset = CRAFTSBDataset(images_dir="/data/sb_craft_dataset/ch4_training_images")

    for i, sample in enumerate(dataset):
        image_path, sb, annotations = sample

        for annotation in annotations:
            xtl = int(annotation[0])
            ytl = int(annotation[1])
            xbr = int(annotation[4])
            ybr = int(annotation[5])
            text = annotation[-1]
            sb_with_box = func_draw_boxes(sb, (xtl, ytl, xbr, ybr))

        # uncomment the line below to save the images with bboxes
        sb_with_box.save(f"sb{i}.jpg")

        if i >= 5:
            break

    print("Done!")
