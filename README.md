# bookgen
generate a book for your learning
- verison: V0.1

# 方法論手動操作
我習慣使用 obsidian 來編輯 md, 並複製這個範本後開始與繼續
- [AIBooks-template](./AIBooks-template.md)

## 手動操作範例書
- [《Vibe Coding：資深工程師與架構師的 LLM 開發新典範》](sample_book/《Vibe%20Coding：資深工程師與架構師的%20LLM%20開發新典範》.md)

## 夥伴試用經驗
- [《從零開始的生物統計之旅：掌握數據、解讀生命》](sample_book/《從零開始的生物統計之旅：掌握數據、解讀生命》.md)

# 使用方法
- 產生的書有兩種：
    - 章節目錄含摘要版在 upload_files, 主要拿來初步閱讀與自動/手動生成小節時使用
    - 完整版（含指定生成過小節）在 output_path
- 生成階段
    - 將生成所需要的檔案（檔名需 .md），放入 upload_files 目錄
    - 修改 config.yaml 
    - 使用 bookgen.ipynb 循序執行，詳細步驟請參考內部說明


# 建構過程說明
- 跟 gemini 對話，設計修改規格文件到規格幾乎差不多。然後整體生成程式。
- 生成後 LLM 改物件化後才開始實作
- 匯入 notebook, .py 到 IDE , 設置環境後測試
- 初步程式碼大致到位，但細部修整還是得除錯不少功夫
    - 倒不是程式碼有 bug, 主要是細部需求畢竟沒全部提供，改的時候就不會整個重新生成。
    - 主要修改 gemini 生成 function, 存檔後再取回, 按格式檔案生成 這幾個部分
- 後面的規格文件由 LLM 生成，還沒仔細改成最符合目前系統實作的樣子，比較像是依照這個規格，生成的起始樣態後修改
   
## AIQA: LLM 輔助書籍生成程式 - 規格文件 **

**1. 專案名稱:** 自學書籍生成程式

**2. 專案目標:**

開發一個基於 Jupyter Notebook 和 YAML 配置的程式，專為 Gemini 模型設計，協助使用者快速生成兩層書籍大綱及章節內容，具備資料存檔與讀取功能，可持久化程式狀態，支援從指定目錄上傳檔案 (預設為 .md 檔) 作為 Gemini 推論的上下文， 並提供彈性的匯出功能與詳盡的 Prompt 紀錄. 程式內部資料格式統一使用 JSON，最終書籍產出格式為 Markdown (.md)，方便工程使用與續用。

**3. 目標使用者:**

* 具備程式設計基礎，希望自行修改和擴充程式碼的使用者
* 偏好使用 Jupyter Notebook 進行開發和實驗的使用者
* 主要使用 Gemini 模型，需要快速生成結構化學習材料的使用者
* 需要詳細的 Prompt 紀錄，方便手動調整或重新生成內容的使用者
* 希望能夠儲存和讀取程式執行狀態，避免重複生成的使用者
* 偏好程式內部資料使用 JSON 格式，最終書籍產出使用 Markdown 格式的使用者
* 需要能上傳本地檔案目錄，作為 Gemini 模型生成內容參考的使用者，預設上傳 Markdown 檔案

**4. 功能需求:**

**4.1. 初始化設定 (YAML 配置文件):**

* **主題設定:**  在 YAML 配置文件中設定想要學習的領域或主題名稱。
* **偏好設定:** 在 YAML 配置文件中設定關於書籍風格、目標讀者、特定疑問或關注點等偏好資訊。
* **LLM 模型設定:**  明確指定使用 Gemini 模型，並設定 Gemini API 金鑰。
* **輸出設定:** 在 YAML 配置文件中設定輸出檔案的儲存路徑。 程式狀態儲存檔案路徑及名稱。
* **提示詞模板設定:** 在 YAML 配置文件中設定提示詞模板. 提示詞模板需支援置入上傳檔案內容的預留位置。
* **檔案上傳設定:**
    * 啟用/停用檔案上傳選項。
    * 指定本地檔案目錄路徑，程式將會讀取此目錄下所有檔案 (預設為 .md 檔案，可擴充設定支援檔案類型)。

**4.2. 程式狀態存檔與讀取:**

* **狀態存檔:**
    * 程式應能將 完整的書籍結構 (包含章節目錄、摘要、已生成的章節內容，皆以 JSON 格式儲存)、Prompt 紀錄 (JSON 格式)、 檔案上傳設定與已讀取的檔案內容 (JSON 格式)** 等程式運行狀態，儲存到指定的 JSON 檔案。
    * 存檔時機：
        * 在完成檔案上傳 (若啟用) 後自動存檔。
        * 在生成兩層大綱並確認後自動存檔。
        * 在每次生成新的章節內容後自動存檔。
        * 提供手動存檔功能 (例如 Notebook Cell 指令)。
* **狀態讀取:**
    * 程式啟動時，自動檢查是否存在已儲存的狀態 JSON 檔案。
    * 若存在，則讀取狀態 JSON 檔案，將已儲存的書籍結構、內容、Prompt 紀錄、檔案上傳設定與已讀取的檔案內容等載入到程式中。
    * 若不存在，則從頭開始流程 (包含檔案上傳，若啟用)。
* **資料結構持久化:**  程式內部管理章節目錄、章節內容、Prompt 紀錄、檔案上傳資訊等資料結構 (例如字典、樹狀結構) 需要設計成易於序列化 (儲存為 JSON) 和反序列化 (從 JSON 讀取) 的格式。

**4.3. 檔案上傳與內容讀取:**

* 目錄讀取: 程式根據 YAML 設定的檔案目錄路徑，讀取目錄下所有支援檔案類型 (預設為 .md) 的檔案。
* 內容讀取: 程式讀取每個檔案的內容，並儲存於程式狀態中 (JSON 格式)。
* 錯誤處理:  程式需處理檔案目錄不存在、檔案讀取失敗等錯誤情況，並提供友善的錯誤訊息。
* 使用者告知: 在 Jupyter Notebook 中顯示檔案上傳與讀取的進度與結果。

**4.4. 書籍大綱生成與確認 (兩層結構):**

* **章節目錄請求:**  程式讀取 YAML 設定，根據主題、偏好 以及已上傳的檔案內容 (若啟用檔案上傳)，使用預設或使用者自訂的提示詞 (Prompt) 向 Gemini 請求書籍的 兩層章節目錄 (章與節)，並要求提供章與節的簡短摘要。
    * 範例 Prompt (包含檔案內容):  `[使用者設定提示詞模板，其中包含置入 [UPLOADED_FILE_CONTENT] 預留位置]`，程式會將讀取的檔案內容字串取代 `[UPLOADED_FILE_CONTENT]` 後，再發送給 Gemini。
    * 範例 Prompt (無檔案內容):  `請藉由一本書來教我 [使用者設定主題]，請先給我兩層（每個章節含小節）的目錄，並對章與各小節，提供很簡短的摘要。我需要全部的章節。`
* **大綱呈現與確認:** 程式在 Jupyter Notebook 中清晰呈現生成的兩層書籍大綱 (包含書名、章節標題、節標題及摘要)。 使用者需在 Notebook 中確認生成的大綱是否符合預期。
* **大綱儲存 (同時存檔程式狀態):** 將確認後的大綱以 JSON 格式 儲存，並同時將包含大綱與檔案上傳資訊的完整程式狀態儲存到狀態 JSON 檔案。

**4.5. 章節內容生成 (按需生成):**

* **章節/節點選擇:** 使用者在 Jupyter Notebook 中指定想要深入了解的章節編號或節點路徑 (例如 1, 或 1.2)。
* **詳細內容請求:** 程式根據使用者選擇的章節或節點，以及已上傳的檔案內容 (若啟用檔案上傳)，自動生成提示詞向 Gemini 請求更詳細的內容說明。
* **內容呈現與儲存 (同時存檔程式狀態):** 程式將 Gemini 回覆的詳細內容呈現於 Jupyter Notebook 中，並將內容以 JSON 格式 儲存到檔案，檔案名稱應包含章節/節點資訊。 程式內部以 JSON 格式管理章節內容，但在最終匯出書籍時，會將章節內容轉換為 Markdown 格式。
* **階層式內容管理:** 與之前規格相同。
* **內容載入:**  程式在請求生成內容前，檢查是否已存在對應的已儲存檔案，若存在則載入 已儲存的程式狀態 (JSON 檔案)，包含已生成的內容 (JSON 格式) 與檔案上傳資訊。

**4.6. 書籍匯出:**

* **層級選擇匯出:**  允許使用者選擇匯出的書籍內容層級：
    * **僅匯出兩層大綱:**  只匯出章節目錄與摘要 (匯出為 Markdown 格式)。
    * **匯出大綱與選定章節詳細內容:** 匯出完整大綱，並包含使用者選定章節 (及其子節點) 的詳細內容 (全部匯出為 Markdown 格式)。
    * **匯出全部已生成內容:** 匯出大綱以及所有已生成的章節與節點詳細內容 (全部匯出為 Markdown 格式)。
* **匯出格式:**  最終書籍一律匯出為 Markdown (.md) 格式。

**4.7. Prompt 紀錄:**

* **詳細 Prompt 紀錄:**  Prompt 紀錄需包含是否使用了上傳檔案內容，以及使用的檔案列表。
* **Prompt 紀錄儲存 (同時存檔程式狀態):** 將 Prompt 紀錄以 JSON 格式 儲存到獨立檔案，並同時將 Prompt 紀錄包含在程式狀態 JSON 檔案中一起儲存。

**5. 非功能需求:**

* **易用性 (針對程式設計使用者):**  程式碼簡潔易懂，Jupyter Notebook 介面方便操作。
* **可維護性與彈性:** 程式碼模組化，YAML 配置參數化，易於修改和擴充。
* **效率:**  快速與 Gemini 交互，避免重複生成，提供快速學習流程。 透過狀態存檔與讀取，進一步提升效率，避免重複工作。 檔案上傳功能應盡可能高效，並支援處理一定數量的檔案。
* **錯誤處理:**  完善的錯誤處理機制，需考慮狀態 JSON 檔案讀寫錯誤、檔案上傳與讀取錯誤、Gemini API 錯誤等。

**6. 輸入與輸出:**

* **輸入:**
    * YAML 配置文件 (主題、偏好、Gemini API 金鑰、輸出設定、提示詞模板、狀態檔案路徑、檔案上傳設定、檔案目錄路徑)
    * 使用者在 Jupyter Notebook 中輸入的章節/節點編號
    * 使用者選擇的匯出層級
* **輸出:**
    * 書籍大綱與摘要 (JSON 檔案)
    * 章節/節點詳細內容 (JSON 檔案，程式內部格式)
    * Prompt 紀錄檔 (JSON 檔案)
    * 程式狀態檔案 (JSON 檔案)
    * Jupyter Notebook 中的程式執行訊息和輸出結果
    * 匯出的書籍內容檔案 (Markdown .md 格式)

**7. 使用者介面 (UI):**

* **Jupyter Notebook 介面:**  主要操作介面。

**8. 技術規格:**

* **程式語言:** Python
* **LLM API 函式庫:**  `google-generativeai` (for Gemini)
* **YAML 函式庫:**  `PyYAML`
* **JSON 函式庫:**  `json` (Python 內建)
* **Markdown 函式庫 (用於匯出):**  例如 `markdown` (Python-Markdown)
* **檔案操作函式庫:**  Python 內建的 `os` 模組

* **檔案格式:**
    * 配置文件：YAML
    * 大綱與摘要：JSON
    * 章節內容：JSON (程式內部格式)
    * Prompt 紀錄：JSON
    * 程式狀態：JSON
    * 最終書籍輸出：Markdown (.md)
    * 預設上傳檔案：Markdown (.md)

**9. 開發流程 (建議):**

1. **需求確認與規格文件最終審核:** 再次確認此規格文件是否完全符合需求。
2. **原型設計與開發 (Jupyter Notebook + Gemini API + 狀態存檔讀取 + JSON 格式 + 檔案上傳 MD):**  開發 Jupyter Notebook 程式碼，實現核心功能 (讀取 YAML 設定、Gemini API 互動、兩層大綱生成、章節內容請求、檔案儲存、Prompt 紀錄、程式狀態存檔與讀取，並確保程式內部資料以 JSON 格式處理，以及實作檔案上傳與讀取功能，預設支援 .md 檔案)。
3. **功能測試與迭代 (Gemini 模型 + 狀態持久化 + JSON 格式 + 檔案上傳 MD 測試):**  使用 Gemini 模型進行功能測試，重點測試狀態存檔讀取功能、JSON 資料格式處理、檔案上傳功能 (特別是 .md 檔案支援)，確保所有功能運作正常且資料持久化正確。
4. **YAML 配置與匯出功能完善 (Markdown 匯出 + 檔案上傳配置 + MD 預設):**  完善 YAML 配置，實現不同層級的書籍匯出功能，實作 Markdown 匯出，將檔案上傳相關設定納入 YAML 配置，並確保預設檔案類型為 .md。
5. **錯誤處理與檔案管理加強:**  加強錯誤處理，完善檔案管理、Prompt 紀錄和狀態 JSON 檔案讀寫錯誤、檔案上傳與讀取錯誤處理，並針對 .md 檔案處理進行測試。
6. **程式碼整理、註解與文件撰寫:**  程式碼整理，加入註解，撰寫詳細使用說明文件，包含檔案上傳功能的使用說明，並特別註明預設支援 .md 檔案。

**10. 後續擴充方向 (使用者可自行擴充):**

* 更精細的 Gemini Prompt 工程。
* 支援更多匯出格式 (HTML, PDF 等)。
* 整合其他 Python 函式庫。
* 提供更精細的狀態管理功能。
* 優化 Markdown 匯出格式。
* 支援更多檔案類型上傳 (例如 PDF, Word 等)，並處理不同檔案類型的內容讀取。
* 提供更彈性的檔案內容使用方式，例如指定檔案內容在 Prompt 中的位置或格式。
* 允許使用者在 YAML 設定中自訂預設上傳檔案類型。
