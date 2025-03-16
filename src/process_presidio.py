from presidio_analyzer import AnalyzerEngine
import re
def load_analyzer():
    # Set up the engine, loads the NLP module (spaCy model by default) and other PII recognizers
    analyzer = AnalyzerEngine()
    return analyzer
def process_results(results,text):
    """
    Process the results from the analyzer to extract the PII entities from the text
    :param results: List of results from the analyzer
    :param text: Text to extract entities from
    :return: List of sensitive entities
    """
    start_list = [i.start for i in results]
    end_list = [i.end for i in results]
    return [text[start_list[i]:end_list[i]].strip() for i in range(len(start_list))]
def detect(text,analyzer,lang="en"):
    """
    Predict the sensitive entities in the text
    :param text: Text to predict entities from
    :param analyzer: Analyzer engine to use
    :param lang: Language of the text
    :return: List of sensitive entities
    """
    results = analyzer.analyze(text=text,
                           language=lang)
    return process_results(results,text)
def process_excel_results(results:list):
    
    #replace | with ""
    results = [result.replace("|","") for result in results]
    results = [result.strip() for result in results]
    results = [result for result in results if result!=""]
    
    # Split results with multiple spaces
    new_results = []
    for result in results:
        # Split by 2 or more consecutive spaces using regex
        parts = re.split(r'\s{2,}', result)
        # Add non-empty parts to new results
        for part in parts:
            part = part.strip()
            if part:  # Ensure part is not empty
                new_results.append(part)
    
    # Return unique results
    return list(set(new_results))
if __name__ == "__main__":
    # Load the analyzer
    analyzer = load_analyzer()
    # Test the analyzer
    text = open("Final Sec Attendee List.md").read()

    #text = "John Doe was born in 1990 and lives in New York."
    entities = detect(text,analyzer)
    entities = process_excel_results(entities)
    with open("pii_entities.txt","w") as f:
        f.write("\n".join(entities))