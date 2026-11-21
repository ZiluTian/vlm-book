"""Typeset chapter conventions at build time; keep GitHub Markdown readable."""
import re


def on_page_markdown(markdown, page, **kwargs):
    if not re.match(r'ch\d{2}-', page.file.src_uri):
        return markdown

    def checkpoint(match):
        text = ' '.join(match[1].splitlines())
        parts = re.split(r'\s*→\s*(?=\*)', text, maxsplit=1)
        if len(parts) != 2:
            return match[0]
        return ('!!! question "Checkpoint"\n    ' + parts[0].strip() +
                '\n\n    ??? success "Reveal the answer"\n        ' + parts[1].strip())

    # Leave fenced examples untouched (including their indentation and newlines).
    chunks = re.split(r'(^```[^\n]*\n[\s\S]*?^```[ \t]*$)', markdown, flags=re.M)
    for i in range(0, len(chunks), 2):
        s = re.sub(r'^> 🎯 \*\*The one sentence:\*\* (.*(?:\n> [^\n]*)*)',
                   lambda m: '!!! abstract "Chapter brief"\n    ' + m[1].replace('\n> ', '\n    '), chunks[i], flags=re.M)
        s = re.sub(r'^\*\*Check:\*\* ([^\n]+(?:\n(?!\s*$|#|\*\*Check:)[^\n]+)*)', checkpoint, s, flags=re.M)
        s = re.sub(r'^(?:🔗 \*\*BEV bridge:\*\*|\*\*🔗 BEV bridge:\*\*) ([^\n]+(?:\n(?!\s*$)[^\n]+)*)',
                   lambda m: '!!! tip "BEV bridge"\n    ' + m[1].replace('\n', '\n    '), s, flags=re.M)
        chunks[i] = s
    return ''.join(chunks)
