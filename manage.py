import os
import sys


def main():
    file_name = sys.argv.pop()
    os.environ.setdefault("SETTINGS_MODULE", f"config.settings.{file_name}")
    try:
        from pipeline.core.management import execute_from_command_line
    except ImportError:
        raise ImportError(
            "Couldn't import pipeline. Are you sure it's installed ?"
        )
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
