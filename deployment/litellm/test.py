# import os 

# from dotenv import load_dotenv

# _ = load_dotenv(".env")
# LITE_LLM_HOST = os.getenv("OPENAI_API_KEY")
# print(f"LITE_LLM_HOST: {LITE_LLM_HOST} hello word")
def clean_env_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        cleaned_line = line.strip().replace("\r", "")  # Xóa \r và khoảng trắng đầu/cuối dòng
        if cleaned_line:  # Bỏ qua dòng trống
            cleaned_lines.append(cleaned_line)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines) + "\n")  # Ghi lại file với các dòng sạch

# Chạy hàm để làm sạch file .env
clean_env_file(".env")
print("File .env đã được làm sạch!")