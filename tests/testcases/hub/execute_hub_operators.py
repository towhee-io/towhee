import os
import sys
sys.path.append("../../../tests")
import yaml
from loguru import logger
from common import common_func as cf

path = '/workspace/towhee_CI/hub_operator'
log_path = path + "/log" + "/hub_op_readme.log"
test_case_path = path + "/testcases"

operator_path = path + '/operator-tasks'

def get_op_clone_address_list_from_yaml(descript_paths):
    """
    Extract the path for a specific file name
    Parameters:
        directory - directory to be extracted
        file_name - file name to be extracted
    Returns:
        a list contains the path for the specified file
        in input directory
    """
    op_link_list = []
    logger.debug(descript_paths)
    for descript_path in descript_paths:
        with open(descript_path, encoding='utf-8')as file:
            content = file.read()
            data = yaml.load(content, Loader=yaml.FullLoader)
            models = data.get("models")
            if models is None:
                continue
            for model in models:
                op_link = list(model.values())[0].get("op-link")
                if op_link not in op_link_list:
                    op_link_list.append(op_link)
            file.close()

    logger.debug("Extracted %d ops \n" % len(op_link_list))
    logger.debug(op_link_list)

    return op_link_list

def extract_python_code_from_op_readme(op_path, op_name):
    """
    Extract the readme of all the ops
    Parameters:
        op_path - the path for operator
        op_name - name of operator
    Returns:
        the test case file
    """
    test_case_name = op_path + "/test_" + op_name + ".py"
    with open(op_path+"/README.md") as f:
        lines = f.readlines()
        f.close()

    lines = [i.strip() for i in lines]
    logger.debug("Readme for %s is \n" % op_path)
    logger.debug("%s" % lines)

    for i in range(len(lines)):
        if ('```Python' in lines[i]) or ('```python' in lines[i]):
            for j in range(i + 1, len(lines)):
                if '```' == lines[j]:
                    break
                else:
                    if "./towhee.jpg" in lines[j]:
                        lines[j] = lines[j].replace('./towhee.jpg', path + '/dataset/test.jpg')
                    elif "./test.png" in lines[j]:
                        lines[j] = lines[j].replace('./test.png', path + '/dataset/test.jpg')
                    elif "./example.jpg" in lines[j]:
                        lines[j] = lines[j].replace('./example.jpg', path + '/dataset/test.jpg')
                    elif "./test_face.jpg" in lines[j]:
                        lines[j] = lines[j].replace('./test_face.jpg', path + '/dataset/test_face.jpg')
                    elif "./towhee.jpeg" in lines[j]:
                        lines[j] = lines[j].replace('./towhee.jpeg', path + '/dataset/test.jpg')
                    elif "turing.png" in lines[j]:
                        lines[j] = lines[j].replace('turing.png', path + '/dataset/test.jpg')
                    elif "./img1.jpg" in lines[j]:
                        lines[j] = lines[j].replace('./img1.jpg', path + '/dataset/test.jpg')
                    elif "./img.png" in lines[j]:
                        lines[j] = lines[j].replace('./img.png', path + '/dataset/test.jpg')
                    elif "./dog.jpg" in lines[j]:
                        lines[j] = lines[j].replace('./dog.jpg', path + '/dataset/test.jpg')
                    elif "./avengers.jpg" in lines[j]:
                        lines[j] = lines[j].replace('./avengers.jpg', path + '/dataset/test.jpg')
                    elif "./teddy.jpg" in lines[j]:
                        lines[j] = lines[j].replace('./teddy.jpg', path + '/dataset/test.jpg')
                    elif "./animals.jpg" in lines[j]:
                        lines[j] = lines[j].replace('./animals.jpg', path + '/dataset/test.jpg')
                    elif "./hulk.jpg" in lines[j]:
                        lines[j] = lines[j].replace('./hulk.jpg', path + '/dataset/test.jpg')
                    elif "./jumpingjack.gif" in lines[j]:
                        lines[j] = lines[j].replace('./jumpingjack.gif', path + '/dataset/test.mp4')
                    elif "./demo_video.mp4" in lines[j]:
                        lines[j] = lines[j].replace('./demo_video.mp4', path + '/dataset/test.mp4')
                    elif  "./archery.mp4" in lines[j]:
                        lines[j] = lines[j].replace('./archery.mp4', path + '/dataset/test.mp4')
                    elif "test.wav" in lines[j]:
                        lines[j] = lines[j].replace('test.wav', path + '/dataset/test.wav')
                    with open(test_case_name, 'a') as f1:
                        f1.write(lines[j] + '\n')
                        f1.close()
    if not os.path.exists(test_case_path):
        os.system(f"mkdir -p {test_case_path}")
    os.system(f"cp {test_case_name} {test_case_path}")
    if os.path.exists(test_case_name):
        logger.debug("Success to create test case %s " % test_case_name)
    else:
        logger.error("Fail to create test case %s " % test_case_name)

    return test_case_name

def execute_ops(op_clone_list):
    """
    Extract the readme of all the ops
    Parameters:
        op_clone_list - clone address for op
    Returns:
        1. install requirements first
        2. executed extracted readme
    """
    success = 0
    op_id = 1
    op_fail_list = []
    for op_address in op_clone_list:
        logger.debug("Running operator %s" % op_address)
        #extract each operator name and its classification
        others, op_name = op_address.rsplit('/', 1)
        logger.debug("Operator name is: %s" % op_name)
        other, classification = others.rsplit('/', 1)
        op_name_single = classification + '-' + op_name
        logger.debug("Operator name with its classification is: %s" % op_name_single)
        op_single_path = path + '/' + op_name_single
        if not os.path.exists(op_single_path):
            os.system(f"mkdir -p {op_single_path}")
        # os.system(f"cd {op_single_path} && git clone {op_address} {op_name_single}")
        op_readme_path_name = op_address + '/raw/branch/main/README.md'
        op_requirement_path_name = op_address + '/raw/branch/main/requirements.txt'
        os.system(f"cd {op_single_path} && wget {op_readme_path_name}")
        os.system(f"cd {op_single_path} && wget {op_requirement_path_name}")
        # op_name_path = path + f"/{op_name_single}"
        # extract python code from op readme
        test_case_name = extract_python_code_from_op_readme(op_single_path, op_name_single)
        if not os.path.exists(test_case_name):
            logger.error("Fail to run readme for %d: operator %s" % (op_id, op_address))
            continue
        op_requirement = op_single_path + "/requirements.txt"
        logger.debug("Install requirements for %s" % op_single_path)
        os.system(f"pip install -r {op_requirement}")
        # execute python code from op readme
        logger.debug("Running extracted readme for %s" % op_single_path)
        test_case_name_backup = test_case_path + "/test_" + op_name_single + ".py"
        res = os.system(f"python3 {test_case_name_backup} >> {log_path}")
        if 0 != res:
            logger.error("Fail to run readme for %d: operator %s" % (op_id, op_address))
            op_fail_list.append(op_address)
        else:
            logger.info("Success to run readme for %d: operator %s" % (op_id,op_address))
            success += 1
        op_id += 1

    if success != len(op_clone_list):
        logger.error("Fail to run readme for [%d] operators: [%s]" % (len(op_fail_list), op_fail_list))
        assert False
    else:
        logger.info("Success to run readme for all operators")



if __name__ == '__main__':

    if not os.path.exists(path):
        os.system(f"mkdir -p {path}")
    res = 0
    operator_tasks_path = path + '/operator-tasks'
    if not os.path.exists(operator_tasks_path):
        res = os.system(f"cd {path} && git clone git@github.com:towhee-io/operator-tasks.git")
    if 0 != res:
        logger.error("Fail to clone operator tasks, please check and retry")
    else:
        logger.info("Success to clone operator tasks repo")
        logger.add(log_path)
        descript_paths = cf.get_specific_file_path(operator_path, "description.yaml")
        op_clone_list = get_op_clone_address_list_from_yaml(descript_paths)
        execute_ops(op_clone_list)
