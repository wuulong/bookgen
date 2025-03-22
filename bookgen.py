# README: bookgen.py 是一個用於生成書籍大綱和內容的 Python 類別。
# verison: V0.1
import yaml
import json
import os
from google import genai
from google.genai import types
import markdown  # 確保您已安裝 python-markdown: pip install markdown
import re

def handle_regex(regex,file_path,type="col2"): 
    """
    return lines
    """
    print(f"parse file_path:{file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
            test_str = file.read()
    #print(test_str)
    matches = re.finditer(regex, test_str, re.DOTALL) #re.MULTILINE
    lines = []
    if type=="col2": #regex="\*\*Q：\*\*(.*)\n\*\*A：\*\*(.*)\n"
        for matchNum, match in enumerate(matches, start=1):
            for groupNum in range(0, len(match.groups())):
                groupNum = groupNum + 1
                mark = "Q" if groupNum==1 else "A"
                group = match.group(groupNum).replace("*","")
                print_str = f"{mark}:{group}"
                print(print_str ) 
                lines.append(print_str)
    if type=="col1":
        
        for matchNum, match in enumerate(matches, start=1):
            
            for groupNum in range(0, len(match.groups())):
                groupNum = groupNum + 1
                group = match.group(groupNum)
                print_str = f"{group}"
                #print(print_str ) 
                lines.append(print_str)
    return lines  

class BookGen:
    """LLM 輔助書籍生成器物件."""

    def __init__(self, config_file='config.yaml'):
        """初始化 BookGen 物件，載入設定檔和程式狀態."""
        self.config = self._load_config(config_file)
        self.state = self._load_state(self.config)
        self.book_structure = self.state.get('book_structure', {})
        self.prompt_log = self.state.get('prompt_log', [])
        #self.uploaded_file_content = self.state.get('uploaded_file_content', "")
        self.uploaded_files_list = self.state.get('uploaded_files_list', []) # 初始化檔案列表
        self.client = None # 保留 Gemini 客戶端物件
        self.files = [] # 保留上傳的檔案列表
        self.topic = self.config.get('topic', '')

    def _load_config(self, config_file='config.yaml'):
        """載入 YAML 配置文件 (私有方法)."""
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def save_state(self):
        """儲存程式狀態到 JSON 檔案."""
        state_file_path = self.config.get('state_file_path', 'program_state.json')
        #os.makedirs(os.path.dirname(state_file_path), exist_ok=True) # 確保路徑存在
        with open(state_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=4, ensure_ascii=False)
        print(f"程式狀態已儲存至: {state_file_path}")

    def _load_state(self, config):
        """從 JSON 檔案載入程式狀態，如果檔案不存在則返回空的狀態 (私有方法)."""
        state_file_path = config.get('state_file_path', 'program_state.json')
        if os.path.exists(state_file_path):
            try:
                with open(state_file_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    print(f"程式狀態已從 {state_file_path} 載入。")
                    return state
            except json.JSONDecodeError:
                print(f"警告: 狀態檔案 {state_file_path} 格式不正確，將重新開始。")
                return {} # JSON 格式錯誤，返回空狀態
        else:
            print("未找到程式狀態檔案，將從頭開始。")
            return {} # 檔案不存在，返回空狀態
    def generate(self,user_prompt,file_path):
        model = self.config.get('llm_model', 'gemini-2.0-flash-thinking-exp-01-21')

        parts = []
        for file in self.files:
            parts.append(types.Part.from_uri(
                        file_uri=file.uri,
                        mime_type=file.mime_type,
                    ))
        parts.append(types.Part.from_text(text=user_prompt))
        contents = [
            types.Content(
                role="user",
                parts=parts
            )
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=64,
            max_output_tokens=65536,
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text="""請以台灣人的立場，用繁體中文回答"""),
            ],
        )
        
        print(f"Q::{user_prompt}")

        response_text = ""
        
        for chunk in self.client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text
            #print(chunk.text, end="")
        print(f"A::{response_text}")

        # Define the file path
        #file_path = f"{dir_txt}/{law_name}_{file_postfix}.txt"

        # Write the content to the file
        output_path = self.config.get('output_path', 'output')
        if not os.path.isdir(output_path):
            print(f"警告：檔案上傳目錄 '{output_path}' 不存在。請檢查設定檔。")
            return "" # 目錄不存在，返回空字串
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"Q::{user_prompt}\n")
                file.write(f"A::{response_text}")
                print(f"Content written to {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
        return response_text
    def upload_files(self):
        """上傳指定目錄下的 .md 檔案並讀取內容."""
        if not self.config.get('upload_files', False):
            print("檔案上傳功能已停用。")
            return "" # 檔案上傳功能停用，返回空字串

        upload_dir = self.config.get('upload_directory', 'upload_files')
        if not os.path.isdir(upload_dir):
            print(f"警告：檔案上傳目錄 '{upload_dir}' 不存在。請檢查設定檔。")
            return "" # 目錄不存在，返回空字串

        self.files = []
        for filename in os.listdir(upload_dir):
            print(f"看到檔案: {filename}")
            if filename.endswith(".md"): # 預設讀取 .md 檔案
                filepath = os.path.join(upload_dir, filename)
                try:
                    self.files.append(self.client.files.upload(file=filepath,config={'mime_type':"text/markdown"}))
                    print(f"已上傳檔案: {filename}")
                except Exception as e:
                    print(f"上傳檔案 {filename} 失敗: {e}")

        return 
    def setup_client(self):
        api_key=self.config['GEMINI_API_KEY']
        
        self.client = genai.Client(
            api_key = api_key,
        )
        
    def generate_outline(self,regen=False):
        """使用 Gemini 生成書籍大綱."""

        prompt_template = self.config['outline_prompt_template']
        outline_format = self.config.get('outline_format', 'outline_format.txt')
        #把 outline_format 的檔案內容讀入存成 outline_format_content
        outline_format_content = ""
        with open(outline_format, 'r', encoding='utf-8') as f:
            outline_format_content = f.read()  
        prompt = prompt_template.format(preferences=self.config['preferences'],outline_format_content=outline_format_content)

        #print("\n--- 發送給 Gemini 的大綱生成 Prompt ---")
        #print(prompt)

        #response = model.generate_content(prompt)
        file_path = self.config.get('output_path', 'output') + "/outline.txt"
        # 檢查 file_path 是否存在，如果存在且不需要重新生成則直接返回
        if os.path.exists(file_path) and not regen:
            print(f"大綱已存在於 {file_path}，將跳過生成。")
        else:
            outline_text = self.generate(prompt,file_path)
        return 

    def parse_outline(self):
        """解析大綱文字，提取章節和節點."""
        book_structure = {}
        current_chapter = None

        
        file_path = self.config.get('output_path', 'output') + "/outline.txt"

        regex = r"A::.*```markdown\n書名：(.*?)\n"    
        lines = handle_regex(regex,file_path,"col1")
        self.topic = "\n".join(lines)
        print(f"topic:{self.topic}")


        regex = r"Q::(.*?)```markdown"    
        lines = handle_regex(regex,file_path,"col1")
        outline_prompt = "\n".join(lines)
        print(f"outline_prompt:{outline_prompt}")

        regex = r"A::.*```markdown\n(.*)```"
        lines = handle_regex(regex,file_path,"col1")
        outline_text = "\n".join(lines)
        print(f"outline length:{len(outline_text)}")
        #json_string = re.sub(r'//.*', '', json_string)
        #json_string = re.sub(r'# .*', '', json_string)
        #print(json_string)
        #json_object = json.loads(json_string) 

        for line in outline_text.splitlines():
            line = line.strip()
            if not line:
                continue # 忽略空行

            if line.startswith("# "): # 章節標題 (假設章節以 #  開頭)
                chapter_title = line[2:].strip()
                current_chapter = chapter_title
                book_structure[current_chapter] = {'summary': '', 'sections': {}}
            elif line.startswith("## "): # 節點標題 (假設節點以 ## 開頭)
                if current_chapter:
                    section_title = line[3:].strip()
                    book_structure[current_chapter]['sections'][section_title] = {'summary': '', 'content': ''}
            elif current_chapter: # 摘要 (假設摘要在章節或節點標題後)
                if not book_structure[current_chapter]['summary']: # 章節摘要
                    book_structure[current_chapter]['summary'] = line
                elif section_title and not book_structure[current_chapter]['sections'][section_title]['summary']: # 節點摘要
                     book_structure[current_chapter]['sections'][section_title]['summary'] = line
                section_title = None # 清空 section_title 方便判斷章節摘要
        return [book_structure, outline_prompt]

    def generate_section_content(self, index,chapter_title, section_title,regen=False):
        """使用 Gemini 生成章節詳細內容."""

        prompt_template = self.config['section_prompt_template']
        section_format = self.config['section_format']
        section_format_content = ""
        with open(section_format, 'r', encoding='utf-8') as f:
            section_format_content = f.read()  
        prompt = prompt_template.format(chapter_title=chapter_title, section_title=section_title,section_format_content=section_format_content)

        #print(f"\n--- 發送給 Gemini 的 {chapter_title} - {section_title} 內容生成 Prompt ---")
        #print(prompt)

        #response = model.generate_content(prompt)

        file_path = self.config.get('output_path', 'output') + f"/I{index}.txt"

        if os.path.exists(file_path) and not regen:
            print(f"生成檔案已存在於 {file_path}，將跳過生成。")
        else:
            section_content_text = self.generate(prompt,file_path)
        
        #print(f"\n--- Gemini 回覆的 {chapter_title} - {section_title} 內容 ---")
        #print(chapter_content_text)
        return file_path
    def parse_section(self,file_path):

        regex = r"Q::(.*?)```markdown"    
        lines = handle_regex(regex,file_path,"col1")
        prompt = "\n".join(lines)
        print(f"prompt:{prompt}")


        regex = r"A::.*```markdown\n(.*)```"
        lines = handle_regex(regex,file_path,"col1")
        outline_text = "\n".join(lines)
        print(f"outline length:{len(outline_text)}")
        return prompt, outline_text
        
    def export_book_to_markdown(self, export_level='all'):
        """將書籍結構匯出為 Markdown 檔案."""
        if export_level=='all':
            output_path = self.config.get('output_path', 'output')
        else:
            output_path = self.config.get('upload_directory', 'upload_files')
        os.makedirs(output_path, exist_ok=True)
        filepath = os.path.join(output_path, f"{self.config['topic']}_book.md")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {self.topic} \n") # 書籍標題

            for chapter_title, chapter_data in self.book_structure.items():
                f.write(f"\n\n# {chapter_title}") # 章節標題
                f.write(f"\n**摘要:** {chapter_data['summary']}") # 章節摘要

                if export_level in ['all', 'chapters_selected']: # 根據匯出層級判斷是否匯出章節內容
                    for section_title, section_data in chapter_data['sections'].items():
                        f.write(f"\n## {section_title}") # 節點標題
                        f.write(f"\n**摘要:** {section_data['summary']}") # 節點摘要
                        if export_level == 'all': # 完整匯出包含內容
                            content = section_data['content']
                            content = re.sub(f'## {section_title}', '', content)
                            f.write(f"\n{content}") # 節點內容

        print(f"書籍已匯出為 Markdown 檔案: {filepath}")

    def get_chapter_section_titles(self):
        """取得章節與節點標題列表，方便使用者選擇."""
        titles = []
        chapter_index = 0
        for chapter_title, chapter_data in self.book_structure.items():
            titles.append({'type': 'chapter', 'index': chapter_index, 'title': chapter_title})
            section_index = 1
            for section_title, section_data in chapter_data['sections'].items():
                titles.append({'type': 'section', 'index': f"{chapter_index}.{section_index}", 'title': section_title, 'chapter_title': chapter_title})
                section_index += 1
            chapter_index += 1
        return titles


if __name__ == "__main__":
    print("請在 Jupyter Notebook 中使用 BookGen 物件。")