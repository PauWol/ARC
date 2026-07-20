import re


test = """# HELP PLEASE 😅

Hi!!

I'm trying to get this working.

Repository:
https://github.com/example/my-project

Docs:
https://example.com/docs/setup

Current file:
./src/agent/main.py

Another path:
/home/paul/projects/my-project/config/settings.json

This is the command I'm running:

```bash
python main.py --debug --port 8080
```

Output:

```text
INFO Starting...
INFO Loading config...
INFO Loading config...
INFO Loading config...
INFO Connecting...
INFO Connecting...
ERROR Failed to connect to localhost:5432
Traceback (most recent call last):
...
```

I also tried

`python main.py`

and

`python main.py`

and

`python main.py`

Still doesn't work.

The error keeps saying it can't connect to the database.

Could you help me figure out why it can't connect to PostgreSQL?

Thanks!

https://stackoverflow.com/questions/123456

```

After preprocessing for a user query, you would ideally end up with something close to:

```
I'm trying to get this working. The error keeps saying it can't connect to the database. Could you help me figure out why it can't connect to PostgreSQL?
```

Notice what disappeared:

- ✅ Markdown headings
- ✅ URLs
- ✅ File paths
- ✅ Code blocks
- ✅ Inline code
- ✅ Duplicate commands
- ✅ Repeated log lines
- ✅ Greeting/filler ("Hi!!", "Thanks!")
- ✅ Extra whitespace

while keeping the actual **intent** and **problem statement** intact. This is the kind of deterministic cleanup many agent systems perform before any LLM-based reasoning."""

print(test)
print("\n\n\n")
print(extract_user_query(test))
