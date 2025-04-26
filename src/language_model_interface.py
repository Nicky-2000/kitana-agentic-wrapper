import json
import boto3
import time
import re
import logging
import openai
from google import genai
from google.genai import types
from botocore.exceptions import ClientError
from ollama import chat
from ollama import ChatResponse
from src.token_observer import llm_token_observer
import os

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.api_type = os.getenv("API_TYPE", "openai")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.embedding_api_type = os.getenv("EMBEDDING_API_TYPE", "openai")
        self.local_model_name = os.getenv("LOCAL_MODEL_NAME", "gemma3:12b")


class LanguageModelInterface:
    def __init__(self, config):
        self.config = config
        if self.config.api_type == "openai":
            openai.api_key = self.config.openai_api_key
        elif self.config.api_type == "bedrock":
            self.bedrock = boto3.client(service_name="bedrock-runtime",
                                        aws_access_key_id=self.config.aws_access_key_id,
                                        aws_secret_access_key=self.config.aws_secret_access_key,
                                        region_name=self.config.aws_region)
        elif self.config.api_type == "google":
            self.google_client = genai.Client(api_key=self.config.google_api_key)
        elif self.config.api_type == "local":
            pass
        else:
            logger.error(f"Unsupported API type: {self.config.api_type}")
            raise ValueError(f"Unsupported API type: {self.config.api_type}")

    def create_embedding(self, text, task_type="SEMANTIC_SIMILARITY"):
        if self.config.embedding_api_type == "openai" or (self.config.embedding_api_type is None and self.config.api_type == "openai"):
            return self._get_openai_embedding(text)
        elif self.config.embedding_api_type == "bedrock" or (self.config.embedding_api_type is None and self.config.api_type == "bedrock"):
            return self._get_bedrock_embedding(text)
        elif self.config.embedding_api_type == "google" or (self.config.embedding_api_type is None and self.config.api_type == "google"):
            return self._get_google_embedding(text, task_type)
    
    def _get_google_embedding(self, text, task_type="SEMANTIC_SIMILARITY"):
        result = self.google_client.models.embed_content(
            model="models/text-embedding-004",
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type))

        if not result or not result.embeddings:
            logger.error("Empty response from Gemini embedding")
            raise RuntimeError("Empty response from Gemini embedding")
        
        return result.embeddings[0].values


    def _get_openai_embedding(self, text):
        # response = openai.Embedding.create(
        #     input=text,
        #     model="text-embedding-ada-002"
        # )
        # return response['data'][0]['embedding']

        client = openai.OpenAI(api_key=self.config.openai_api_key)

        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )

        embedding = response.data[0].embedding
        return embedding

    def _get_bedrock_embedding(self, text):
        body = json.dumps({
            "inputText": text
        })
        response = self.bedrock.invoke_model(body=body, modelId="amazon.titan-embed-text-v2:0")
        response_body = json.loads(response.get('body').read())
        return response_body['embedding']

    def get_response(self, prompt, step_name, expect_json=True):
        if self.config.api_type == "openai":
            return self._get_openai_response(prompt)
        elif self.config.api_type == "bedrock":
            if expect_json:
                return self._get_bedrock_response_json(prompt, step_name)
            else:
                return self._get_bedrock_response_text(prompt, step_name)
        elif self.config.api_type == "google":
            return self._get_google_response(prompt)
        elif self.config.api_type == "local":
            response = self._get_local_response(prompt)
            if expect_json:
                response = self._gemini_json_cleaner(response)
            return response

    
    def _get_local_response(self, prompt):
        response: ChatResponse = chat(model=self.config.local_model_name, messages=[
          {
            'role': 'user',
            'content': prompt,
          },
        ])
        # Extracts from ollama ChatResponse dictionary/object
        prompt_tokens = response.get('prompt_eval_count', 0) # Ollama names
        completion_tokens = response.get('eval_count', 0)    # Ollama names
        total_tokens = prompt_tokens + completion_tokens # Calculated
        
        # Passes them separately
        llm_token_observer.update_tokens('local',
                                         prompt_tokens=prompt_tokens,
                                         completion_tokens=completion_tokens,
                                         total_tokens=total_tokens)
        return response.message.content


    
    def _get_google_response(self, prompt):
        response = self.google_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        if not response:
            return {"error": "Empty response from Gemini"}
        
        usage = response.usage_metadata
        prompt_tokens = getattr(usage, 'prompt_token_count', 0)
        completion_tokens = getattr(usage, 'candidates_token_count', 0) # Google calls output 'candidates'
        total_tokens = getattr(usage, 'total_token_count', 0)

        # Passes them separately
        llm_token_observer.update_tokens('google',
                                 prompt_tokens=prompt_tokens,
                                 completion_tokens=completion_tokens,
                                 total_tokens=total_tokens)
        
        return response.text


    def _get_openai_response(self, prompt):

        client = openai.OpenAI(api_key=self.config.openai_api_key)

        response = client.chat.completions.create(
        #openai.ChatCompletion.create(
            model="gpt-4o-mini", #"gpt-3.5-turbo", #this might be cheaper, idk
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )

        if not response:
            return {"error": "Empty response from OpenAI"}
        
        usage = response.usage
        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        completion_tokens = getattr(usage, 'completion_tokens', 0)
        total_tokens = getattr(usage, 'total_tokens', 0)

        # Passes them separately to the observer
        llm_token_observer.update_tokens('openai',
                                         prompt_tokens=prompt_tokens,
                                         completion_tokens=completion_tokens,
                                         total_tokens=total_tokens)

        content = response.choices[0].message.content.strip()
        #print(f"OpenAI Response: {content}")  # Debugging: See what is returned

        try:
            json_start = content.find('{')
            if json_start != -1: 
                content_fin = content[json_start:]

            return json.loads(content_fin)  # Decode only if it's valid JSON
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "content": content}

        #return json.loads(response.choices[0].message.content)

    def _get_bedrock_response_json(self, prompt, step_name):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._get_response_from_bedrock(prompt)
                #print(f"Response: {response}")
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    if attempt == max_retries - 1:
                        return self.fallback_json_extraction(response, step_name)
                    logger.warning(f"JSON decode error in {step_name} (attempt {attempt + 1}/{max_retries}). Retrying...")
            except Exception as e:
                logger.error(f"Unexpected error in {step_name} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
        
        logger.error(f"Max retries reached for {step_name}. Returning empty dict.")
        return {}

    def _get_bedrock_response_text(self, prompt, step_name):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self._get_response_from_bedrock(prompt)
            except Exception as e:
                logger.error(f"Unexpected error in {step_name} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
        
        logger.error(f"Max retries reached for {step_name}. Returning empty string.")
        return ""

    def _get_response_from_bedrock(self, content):
        body = json.dumps({
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": content}],
            "anthropic_version": "bedrock-2023-05-31"
        })

        while True:
            try:
                response = self.bedrock.invoke_model(body=body, modelId="anthropic.claude-3-5-sonnet-20240620-v1:0")
                response_body = json.loads(response.get("body").read())
                # Extracts from response_body['usage'] dictionary
                usage_info = response_body.get("usage", {})
                prompt_tokens = usage_info.get("input_tokens", 0)    # Bedrock/Anthropic names
                completion_tokens = usage_info.get("output_tokens", 0) # Bedrock/Anthropic names
                total_tokens = prompt_tokens + completion_tokens # Usually calculated

                # Passes them separately
                llm_token_observer.update_tokens('bedrock',
                                                 prompt_tokens=prompt_tokens,
                                                 completion_tokens=completion_tokens,
                                                 total_tokens=total_tokens)
                res_content = response_body.get("content")[0]['text']
                return res_content.strip()
            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException':
                    logger.warning("Rate limit exceeded, retrying...")
                    match = re.findall(r"\b(\d+) seconds\b", str(e))
                    if match:
                        sleep_time = int(match[0])
                        logger.error(f"Error: {e}")
                        logger.info(f"Retrying to get response after {sleep_time} seconds")
                        time.sleep(sleep_time)
                    else:
                        logger.error("Could not extract sleep time from error message.")
                        time.sleep(5)  # Default sleep time if we can't parse the message
                else:
                    logger.error(f"Unexpected error: {e}")
                    raise e

    def fallback_json_extraction(self, response, step_name):
        logger.warning(f"Failed to parse JSON in {step_name}. Attempting fallback extraction.")
        try:
            # Try to extract JSON-like structure from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                logger.error(f"No JSON-like structure found in {step_name} response.")
                return {}
        except json.JSONDecodeError:
            logger.error(f"Fallback JSON extraction failed in {step_name}.")
            return {}
        

    def get_text_response(self, prompt):
        if self.config.api_type == "openai":
            return self._get_openai_text_response(prompt)
        elif self.config.api_type == "bedrock":
            return NotImplementedError
        elif self.config.api_type == "google":
            return self._get_google_response(prompt)
        elif self.config.api_type == "local":
            return self._get_local_response(prompt)
        
    def _get_openai_text_response(self, prompt):

        client = openai.OpenAI(api_key=self.config.openai_api_key)

        response = client.chat.completions.create(
        #openai.ChatCompletion.create(
            model="gpt-4o-mini", #"gpt-3.5-turbo", #this might be cheaper, idk
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )

        if not response:
            return {"error": "Empty response from OpenAI"}

        # Extracts from response.usage object
        usage = response.usage
        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        completion_tokens = getattr(usage, 'completion_tokens', 0)
        total_tokens = getattr(usage, 'total_tokens', 0)

        # Passes them separately to the observer
        llm_token_observer.update_tokens('openai',
                                         prompt_tokens=prompt_tokens,
                                         completion_tokens=completion_tokens,
                                         total_tokens=total_tokens)

        content = response.choices[0].message.content.strip()
        return content


    def _gemini_json_cleaner(self, output:str):
        """
        clean common formatting issues with gemini output
        """
        start_index = 0
        end_index = len(output)
        for i in range(len(output)):
            if output[i] == '{':
                start_index = i
                break
        for i in range(len(output)-1, -1, -1):
            if output[i] == '}':
                end_index = i + 1
                break
        return output[start_index:end_index]
