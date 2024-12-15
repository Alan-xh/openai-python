# OpenAI Python 库

OpenAI Python 库提供从用Python编写的应用程序方便访问OpenAI API的功能。它包含了一系列预定义的类，用于API资源，这些类可以根据API响应动态初始化，这使得它与广泛版本的OpenAI API兼容。

您可以在我们的[API参考](https://beta.openai.com/docs/api-reference?lang=python)和[OpenAI Cookbook](https://github.com/openai/openai-cookbook/)中找到OpenAI Python库的使用示例。

## 安装

除非您想修改包，否则无需此源代码。如果您只想使用包，只需运行：

```sh
pip install --upgrade openai
```

从源代码安装：

```sh
python setup.py install
```

### 可选依赖

安装[`openai.embeddings_utils`](openai/embeddings_utils.py)的依赖：

```sh
pip install openai[embeddings]
```

安装[Weights & Biases](https://wandb.me/openai-docs)支持：

```sh
pip install openai[wandb]
```

由于其大小，数据库如`numpy`和`pandas`默认情况下不会安装。它们对于此库的某些功能是需要的，但通常不是与API通信所必需的。如果您遇到`MissingDependencyError`，请使用以下命令安装：

```sh
pip install openai[datalib]
```

## 使用

库需要使用您的账户密钥进行配置，该密钥可以在[网站](https://platform.openai.com/account/api-keys)上找到。可以设置为环境变量`OPENAI_API_KEY`：

```bash
export OPENAI_API_KEY='sk-...'
```

或者将`openai.api_key`设置为其值：

```python
import openai
openai.api_key = "sk-..."

# 列出模型
models = openai.Model.list()

# 打印第一个模型的ID
print(models.data[0].id)

# 创建一个完成
completion = openai.Completion.create(model="ada", prompt="Hello world")

# 打印完成内容
print(completion.choices[0].text)
```

### 参数
所有端点都有一个`.create`方法，支持`request_timeout`参数。此参数接受一个`Union[float, Tuple[float, float]]`，如果请求超出该时间（以秒为单位），将引发`openai.error.Timeout`错误（见：https://requests.readthedocs.io/en/latest/user/quickstart/#timeouts）。

### Microsoft Azure 端点

为了使用Microsoft Azure端点，您需要额外设置`api_type`、`api_base`和`api_version`，除了`api_key`。`api_type`必须设置为'azure'，其他参数对应您端点的属性。此外，部署名称必须作为engine参数传递。

```python
import openai
openai.api_type = "azure"
openai.api_key = "..."
openai.api_base = "https://example-endpoint.openai.azure.com"
openai.api_version = "2023-03-15-preview"

# 创建一个完成
completion = openai.Completion.create(deployment_id="deployment-name", prompt="Hello world")

# 打印完成内容
print(completion.choices[0].text)
```

请注意，目前，Microsoft Azure端点只能用于完成、嵌入和微调操作。
有关如何使用Azure端点进行微调和其他操作的详细示例，请查看以下Jupyter笔记本：
* [使用Azure完成](https://github.com/openai/openai-cookbook/tree/main/examples/azure/completions.ipynb)
* [使用Azure微调](https://github.com/openai/openai-cookbook/tree/main/examples/azure/finetuning.ipynb)
* [使用Azure嵌入](https://github.com/openai/openai-cookbook/blob/main/examples/azure/embeddings.ipynb)

### Microsoft Azure Active Directory 认证

为了使用Microsoft Active Directory来认证您的Azure端点，您需要将`api_type`设置为"azure_ad"，并将获取的凭证令牌传递给`api_key`。其余参数需要按照上一节中指定的设置。

```python
from azure.identity import DefaultAzureCredential
import openai

# 请求凭证
default_credential = DefaultAzureCredential()
token = default_credential.get_token("https://cognitiveservices.azure.com/.default")

# 设置参数
openai.api_type = "azure_ad"
openai.api_key = token.token
openai.api_base = "https://example-endpoint.openai.azure.com/"
openai.api_version = "2023-03-15-preview"

# ...
```

### 命令行界面

此库还提供了一个`openai`命令行工具，使得从终端与API交互变得简单。运行`openai api -h`查看用法。

```sh
# 列出模型
openai api models.list

# 创建一个完成
openai api completions.create -m ada -p "Hello world"

# 创建一个对话完成
openai api chat_completions.create -m gpt-3.5-turbo -g user "Hello world"

# 通过DALL·E API生成图像
openai api image.create -p "two dogs playing chess, cartoon" -n 1
```

## 示例代码

如何使用此Python库完成各种任务的示例可以在[OpenAI Cookbook](https://github.com/openai/openai-cookbook/)中找到。它包含以下代码示例：

* 使用微调进行分类
* 聚类
* 代码搜索
* 自定义嵌入
* 从文档语料库中回答问题
* 推荐
* 嵌入可视化
* 等等

在2022年7月之前，此OpenAI Python库的示例文件夹中托管了代码示例，但自那以后所有示例都已迁移到[OpenAI Cookbook](https://github.com/openai/openai-cookbook/)。

### 聊天

像`gpt-3.5-turbo`这样的对话模型可以通过对话完成端点调用。

```python
import openai
openai.api_key = "sk-..."  # 以您选择的方式提供API密钥

completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world!"}])
print(completion.choices[0].message.content)
```

### 嵌入

在OpenAI Python库中，嵌入表示将文本字符串表示为固定长度的浮点数向量。嵌入旨在测量文本字符串之间的相似性或相关性。

要获取一个文本字符串的嵌入，您可以像以下Python示例一样使用嵌入方法：

```python
import openai
openai.api_key = "sk-..."  # 以您选择的方式提供API密钥

# 选择要嵌入的文本
text_string = "sample text"

# 选择一个嵌入模型
model_id = "text-similarity-davinci-001"

# 计算文本的嵌入
embedding = openai.Embedding.create(input=text_string, model=model_id)['data'][0]['embedding']
```

如何调用嵌入方法的示例可以在[获取嵌入笔记本](https://github.com/openai/openai-cookbook/blob/main/examples/Get_embeddings.ipynb)中找到。

使用嵌入的示例在以下Jupyter笔记本中共享：

- [使用嵌入进行分类](https://github.com/openai/openai-cookbook/blob/main/examples/Classification_using_embeddings.ipynb)
- [使用嵌入进行聚类](https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb)
- [使用嵌入进行代码搜索](https://github.com/openai/openai-cookbook/blob/main/examples/Code_search.ipynb)
- [使用嵌入进行语义文本搜索](https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb)
- [用户和产品嵌入](https://github.com/openai/openai-cookbook/blob/main/examples/User_and_product_embeddings.ipynb)
- [使用嵌入进行零样本分类](https://github.com/openai/openai-cookbook/blob/main/examples/Zero-shot_classification_with_embeddings.ipynb)
- [使用嵌入进行推荐](https://github.com/openai/openai-cookbook/blob/main/examples/Recommendation_using_embeddings.ipynb)

有关嵌入和OpenAI提供的嵌入类型更多信息，请阅读[嵌入指南](https://beta.openai.com/docs/guides/embeddings)。

### 微调

通过在训练数据上微调模型，可以同时提高结果（通过为模型提供更多学习示例）并降低API调用的成本/延迟（主要通过减少在提示中包含训练示例的需求）。

微调的示例在以下Jupyter笔记本中共享：

- [使用微调进行分类](https://github.com/openai/openai-cookbook/blob/main/examples/Fine-tuned_classification.ipynb)（一个简单的笔记本，展示了微调所需的步骤）
- 微调一个回答有关2020年奥运会问题的模型
  - [步骤1：收集数据](https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-1-collect-data.ipynb)
  - [步骤2：创建一个合成的Q&A数据集](https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-2-create-qa.ipynb)
  - [步骤3：训练一个专门用于Q&A的微调模型](https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-3-train-qa.ipynb)

将您的微调同步到[Weights & Biases](https://wandb.me/openai-docs)以便在您的中央仪表板上跟踪实验、模型和数据集：

```bash
openai wandb sync
```

有关微调的更多信息，请阅读[微调指南](https://beta.openai.com/docs/guides/fine-tuning)。

### 内容审核

OpenAI提供了一个Moderation端点，可以用来检查内容是否符合OpenAI [内容政策](https://platform.openai.com/docs/usage-policies)

```python
import openai
openai.api_key = "sk-..."  # 以您选择的方式提供API密钥

moderation_resp = openai.Moderation.create(input="Here is some perfectly innocuous text that follows all OpenAI content policies.")
```

更多详情请见[审核指南](https://platform.openai.com/docs/guides/moderation)。

## 图像生成 (DALL·E)

```python
import openai
openai.api_key = "sk-..."  # 以您选择的方式提供API密钥

image_resp = openai.Image.create(prompt="two dogs playing chess, oil painting", n=4, size="512x512")
```

## 音频转录 (Whisper)
```python
import openai
openai.api_key = "sk-..."  # 以您选择的方式提供API密钥
f = open("path/to/file.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", f)
```

## 异步API

通过在网络绑定方法前添加`a`，可以使用异步支持：

```python
import openai
openai.api_key = "sk-..."  # 以您选择的方式提供API密钥

async def create_completion():
    completion_resp = await openai.Completion.acreate(prompt="This is a test", model="davinci")
```

为了使异步请求更高效，您可以传入您自己的 `aiohttp.ClientSession`，但您必须在程序/事件循环结束时手动关闭客户端会话：

```python
import openai
from aiohttp import ClientSession

openai.aiosession.set(ClientSession())
# 在程序结束时，关闭http会话
await openai.aiosession.get().close()
```

更多详情请见[使用指南](https://platform.openai.com/docs/guides/images)。

## 要求

- Python 3.7.1+

一般来说，我们希望支持我们的客户使用的Python版本。如果您遇到任何版本问题，请在我们的[支持页面](https://help.openai.com/en/)上告知我们。

## 致谢

此库是基于[Stripe Python库](https://github.com/stripe/stripe-python)的分支。