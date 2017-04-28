#!/usr/bin/env python
import os
from os import listdir
from os.path import isfile, join

def onlyFiles(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

class CaffeConfig:
    ERROR_SCENARIO_NOT_SET = "No scenarios dir and/or scenario set!"
    def __init__(self,scenarios_dir=None, scenario=None):
        self.scenarios_dir=scenarios_dir
        self.scenario=scenario
        self.setScenario(scenarios_dir, scenario)

    def _scenario_check(self):
        return self.scenarios_dir != None and self.scenario != None

    def setScenario(self, scenarios_dir, scenario):
        self.scenarios_dir=scenarios_dir
        self.scenario=scenario

        if not(self._scenario_check()):
            return

        self.scen_dir = os.path.join(scenarios_dir, scenario)

    def empty_dir(self):
        self.dir_exists_or_create()

        for the_file in onlyFiles(self.scen_dir):
            file_path = os.path.join(self.scen_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)


    def dir_exists_or_create(self):
        if not self._scenario_check():
            raise Exception(CaffeConfig.ERROR_SCENARIO_NOT_SET)

        # create scenario dir if it not exists
        if not os.path.exists(self.scen_dir):
            try:
                os.makedirs(self.scen_dir)
            except OSError as exc:  # Guard against race condition
                pass

    def path(self):
        return None # implemented in subclasses!

    def save(self, obj):
        self.dir_exists_or_create()
        f = open((self.path()), 'w')
        f.write(str(obj))
        return f.name