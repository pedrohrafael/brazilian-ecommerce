import os
import gradio
from models.predict_model import AnalyzeSentiment

def gradio_wrapper(sentence):
    analizer = AnalyzeSentiment(MODEL)
    result = analizer.predict(sentence)
    return {result['sentiment'] : result['proba']}

MODEL_DIR = '95866330256478990561605bbf7e39d6'
MODEL = os.path.join(os.path.dirname(os.getcwd()), 'models', MODEL_DIR, 'artifacts', 'model.pkl')

demo = gradio.Interface(
    fn=gradio_wrapper,
    inputs=gradio.Textbox(placeholder="Enter a positive or negative sentence here..."), 
    outputs="label",
    interpretation="default",
    title="Analise de Sentimentos",
    description= "Escreva uma avaliação de produto para obter um retorno",
    examples=[["O produto em si tem uma ótima qualidade e acabamento. A entrega foi muito eficiente."]])
    
demo.launch(share=True)