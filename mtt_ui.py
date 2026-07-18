"""
Console-script entry point for `mtt-ui`.

This exists because a pip console-script entry point must point to a plain
Python callable — it can't run an arbitrary shell command like
`streamlit run transcriber_app.py`. transcriber_app.py also isn't safe to
just `import` directly here: it calls st.set_page_config() and friends at
module level, which only work inside a real, live Streamlit script run
(the same reason `python transcriber_app.py` doesn't work either — only
`streamlit run transcriber_app.py` does).

So this wraps Streamlit's own CLI programmatically, which is the documented
way to launch a Streamlit app from inside another script/entry point: it's
equivalent to running `streamlit run transcriber_app.py` yourself, just
callable as a plain function so `[project.scripts]` can point to it.
"""

import os
import sys


def main():
    from streamlit.web import cli as stcli

    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcriber_app.py")

    # Anything passed to `mtt-ui` (e.g. `mtt-ui --server.port 8502`) is
    # forwarded straight through to `streamlit run`, same as it would be if
    # you ran that command yourself.
    sys.argv = ["streamlit", "run", app_path] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
