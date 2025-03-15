import pandas as pd 



xlsx_path = "../documents/sec12_attlist_final.xlsx"
sheet_list = pd.ExcelFile(xlsx_path).sheet_names
for sheet in  sheet_list:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    print(df.head())
    print(df.columns)
    with open(f"{sheet}.md", 'w') as f:
        f.write(df.to_markdown())