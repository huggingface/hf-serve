from pathlib import Path

def get_available_tasks():
    """
    Small hack to retrieve available tasks by scanning the tasks directory.
    This function assumes that each task is defined in a .py file under
    the 'tasks/[library]' directory (ignoring '__init__.py').
    """
    tasks_dir = Path(__file__).parent / "tasks"
    available_tasks = []
    if tasks_dir.exists() and tasks_dir.is_dir():
        for library_dir in tasks_dir.iterdir():
            if library_dir.is_dir():
                # List all .py files (excluding __init__.py) in the library directory.
                # Replace "_" with "-" in the final strings.
                available_tasks.extend(
                    [p.stem.replace("_", "-") for p in library_dir.glob("*.py") if p.name != "__init__.py"]
                )
    return available_tasks
