from render import main as render_main
from pathlib import Path

from multiprocessing import Pool

def render_main_wrap(args):
    #render_main(*args)
    print(*args)


def render_main_wrap2(blender_path, output_dir, obj_path, setting_json_path, script_path):
    print(blender_path, output_dir, obj_path, setting_json_path, script_path)


if __name__ == "__main__":
    # "./data/trans01_rot5deg_fov25_intrinerror_distbig.json"
    settings = Path("./data/settings/").glob("*.json")
    args_list = []
    settings = [Path("./data/settings/distbig_rotnoise.json"), Path("./data/settings/distbig_rotnoise_intrinerror.json")]
    settings = [Path("./data/settings/distbig_rotnoise.json")]
    for setting in settings:
        if setting.stem == "base_setting":
            continue
        print(setting)
        for i in range(90, 91):
            setting_json_path = setting
            blender_path = None
            output_dir = Path("./render/" + Path(setting_json_path).stem + "/" + str(i))
            obj_path = "./data/charuco.obj"
            script_path = "./render_bl.py"

            if output_dir.exists():
                continue
    
            output_dir = str(output_dir)

            #render_main(blender_path, output_dir, obj_path, setting_json_path, script_path)
            args_list.append((blender_path, output_dir, obj_path, setting_json_path, script_path))

    with Pool(8) as p:
        p.starmap(render_main, args_list)
