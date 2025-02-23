from generate_response.models import Base
import os
import time
from utils.utils import read_prompt, local_image_to_data_url
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class Gemini(Base):
    def load_model(self):
        genai.configure(api_key=os.environ['GEMINI_KEY'])
        self.model = genai.GenerativeModel("gemini-1.5-flash-8b")

    def name(self):
        return "gemini"
    
    def send_request(self, text, image_path=None, max_new_tokens=512, temperature=0.0):
        try:
            if image_path is not None:
                image_url, mime_type = local_image_to_data_url(image_path)
                myfile = genai.upload_file(path=image_path, display_name=image_path.split("/")[-1].split(".")[0], mime_type=mime_type)
                message = [myfile, "\n\n", text]
            else:
                message = text
        
            response = self.model.generate_content(
                message,
                generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    # candidate_count=1,
                    # stop_sequences=["x"],
                    max_output_tokens=max_new_tokens,
                    temperature=temperature,
                ),                
            )
            return response.text
        except Exception as e:
            print(e)
            time.sleep(20)
        return ''
        

# gemma_model = Gemma(data_dir="data/memecap", data_type="test", add_image_descriptions=True, add_implications=True)
# gemma_model.load_model()
# gemma_model.load_dataset()
# gemma_model.generate_explanation()
