import os
import json
import shutil
import inspect
config = None

def init(check_imports=None):
    log("Init")

    if check_imports is not None:
        import importlib.util
        for imp in check_imports:
            spec = importlib.util.find_spec(imp)
            if spec is None:
                log(f"{imp} is required, please check requirements are installed.",
                    type="ERROR", always=True)
                return False

    install_js()
    return True

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def log(message, type=None, always=False, name=None):
    if not always and not is_logging_enabled():
        return

    if type is not None:
        message = f"[{type}] {message}"

    if name is None:
        name = get_extension_config()["name"]

    print(f"(pysssss:{name}) {message}")

def is_logging_enabled():
    config = get_extension_config()
    if "logging" not in config:
        return False
    return config["logging"]

def get_extension_config(reload=False):
    global config
    if reload == False and config is not None:
        return config

    config_path = get_ext_dir("nodes.json")
    default_config_path = get_ext_dir("nodes.default.json")
    if not os.path.exists(config_path):
        if os.path.exists(default_config_path):
            shutil.copy(default_config_path, config_path)
            if not os.path.exists(config_path):
                log(f"Failed to create config at {config_path}", type="ERROR", always=True, name="???")
                print(f"Extension path: {get_ext_dir()}")
                return {"name": "Unknown", "version": -1}
    
        else:
            log("Missing pysssss.default.json, this extension may not work correctly. Please reinstall the extension.",
                type="ERROR", always=True, name="???")
            print(f"Extension path: {get_ext_dir()}")
            return {"name": "Unknown", "version": -1}

    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return config

def install_js():
    src_dir = get_ext_dir("web/js")
    if not os.path.exists(src_dir):
        log("No JS")
        return

    should_install = should_install_js()
    if should_install:
        log("it looks like you're running an old version of ComfyUI that requires manual setup of web files, it is recommended you update your installation.", "warning", True)
    dst_dir = get_web_ext_dir()
    linked = os.path.islink(dst_dir) or is_junction(dst_dir)
    if linked or os.path.exists(dst_dir):
        if linked:
            if should_install:
                log("JS already linked")
            else:
                os.unlink(dst_dir)
                log("JS unlinked, PromptServer will serve extension")
        elif not should_install:
            shutil.rmtree(dst_dir)
            log("JS deleted, PromptServer will serve extension")
        return
    
    if not should_install:
        log("JS skipped, PromptServer will serve extension")
        return
    
    if link_js(src_dir, dst_dir):
        log("JS linked")
        return

    log("Copying JS files")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)