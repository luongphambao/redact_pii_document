import pandas as pd 
from process_presidio import load_analyzer,detect,process_excel_results
from visualize_csv import visualize_sheet_data

xlsx_path = "../documents/sec12_attlist_final.xlsx"
sheet_list = pd.ExcelFile(xlsx_path).sheet_names
analyzer = load_analyzer()

for sheet in  sheet_list:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    markdown = df.to_markdown(index = False)
    results = detect(text=markdown,analyzer=analyzer)
    list_sensitive_text = process_excel_results(results)
    html_path, excel_path = visualize_sheet_data(df, list_sensitive_text, sheet)
    print(f"HTML output saved to: {html_path}")
    print(f"Excel output saved to: {excel_path}")
