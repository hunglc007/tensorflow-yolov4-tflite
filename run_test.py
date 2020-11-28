import os


DETECT_SCRIPT = "./scripts/detect.py"
TEST_DIR = "./testdata"
RESULT_DIR = "./testdata_result"


if __name__ == "__main__":
    files = os.listdir(TEST_DIR)

    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    for f in files:
        input_path = f"{TEST_DIR}/{f}"
        output_path = f"{RESULT_DIR}/{f}"
        print(f"running: {input_path}")
        os.system(f"python {DETECT_SCRIPT} --image {input_path} --output {output_path}")
