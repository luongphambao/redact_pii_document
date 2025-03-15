from presidio_analyzer import AnalyzerEngine

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
    return [text[start_list[i]:end_list[i]] for i in range(len(start_list))]
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
