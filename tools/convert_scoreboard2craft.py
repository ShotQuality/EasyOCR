from sq_datasets import ScoreboardDataset
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def get_scoreboard(sample):
    frame = sample["image"]
    sb_bbox = sample["boxes"]["xyx2y2_abs"][
        sample["boxes"]["label"].index("Scoreboard")
    ]
    x1, y1, x2, y2 = sb_bbox
    return frame.crop((x1, y1, x2, y2))


def convert_annotations(image_annotations):
    """
    Convert the annotations from the format used in the scoreboard dataset to the format used in CRAFT
    :param image_annotations: list of annotations for one image
    :return: list of annotations for one image in CRAFT format
    """
    sb_box = image_annotations["xyx2y2_abs"][
        image_annotations["label"].index("Scoreboard")
    ]
    sb_box_origin = [sb_box[0], sb_box[1]]

    annotations = []
    for text, box in zip(image_annotations["text"], image_annotations["xyx2y2_abs"]):
        if text == "":
            continue

        xtl, ytl, xbr, ybr = box
        # since we are croping the scoreboard, subtract the origin of the scoreboard box
        xtl = xtl - sb_box_origin[0]
        xbr = xbr - sb_box_origin[0]
        ytl = ytl - sb_box_origin[1]
        ybr = ybr - sb_box_origin[1]

        box_4coords = [xtl, ytl, xbr, ytl, xbr, ybr, xtl, ybr]
        box_4coords = list(map(lambda x: int(round(x)), box_4coords))

        annotations.append([*box_4coords, text])

    return annotations


def write_annotation_file(craft_annotations, filepath):
    # Open the file in write mode
    with open(filepath, "w") as file:
        for box in craft_annotations:
            # Convert each element in the inner list to a string and join them with commas
            line = ",".join(map(str, box))

            # Write the line to the file
            file.write(line + "\n")


def scoreboard2craft(dataset, images_out_dir="/data/sb_craft_images"):
    # TODO: ugly
    out_images_dir = Path(images_out_dir)
    out_annotations_dir = Path(
        images_out_dir.replace("images", "localization_transcription_gt")
    )

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_annotations_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        annotations = sample["boxes"]
        image_path = Path(sample["image_path"])
        image_name = image_path.name

        # crop scoreboard region and save it
        scoreboard_image = get_scoreboard(sample)
        scoreboard_image.save(str(out_images_dir / image_name))

        # convert annotations and save
        annotations_craft = convert_annotations(annotations)
        write_annotation_file(
            annotations_craft,
            str(out_annotations_dir / f"gt_{image_name.replace('.jpg', '.txt')}"),
        )

        if i >= 5:  # TODO: remove!
            break


if __name__ == "__main__":
    train_dataset = ScoreboardDataset.load_dataset(split="train")
    scoreboard2craft(
        train_dataset, images_out_dir="/data/sb_craft_dataset/ch4_training_images"
    )

    test_dataset = ScoreboardDataset.load_dataset(split="test")
    scoreboard2craft(
        train_dataset, images_out_dir="/data/sb_craft_dataset/ch4_test_images"
    )

    print("end")
