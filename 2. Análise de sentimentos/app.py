import os
import gradio
from models.predict_model import AnalyzeSentiment

MODEL_DIR = '95866330256478990561605bbf7e39d6'
MODEL = os.path.join(os.getcwd(), 'models', MODEL_DIR, 'artifacts', 'model.pkl')

def gradio_wrapper(sentence):
    analizer = AnalyzeSentiment(MODEL)
    result = analizer.predict(sentence)
    return {result['sentiment'] : result['proba']}

def main():
    demo = gradio.Interface(
        fn=gradio_wrapper,
        inputs=gradio.Textbox(placeholder="Digite uma frase positiva ou negativa aqui..."), 
        outputs="label",
        interpretation="default",
        title="Analise de Sentimentos",
        description= "Escreva uma avaliação de produto para obter um retorno",
        examples=[["O produto em si tem uma ótima qualidade e acabamento. A entrega foi muito eficiente."]])

    demo.launch(share=True)
    
if __name__ == "__main__":
    main()
