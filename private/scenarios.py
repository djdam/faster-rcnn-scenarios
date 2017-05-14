#!/usr/bin/env python
from os import chmod
from os.path import join, dirname, abspath
import sys
from copy import deepcopy
import yaml
import scales
import ratios
import ratios_and_scales
import different_config_settings
from default import scenarios_dir

this_dir = dirname(abspath(__file__))
sys.path.insert(0,join(this_dir,'..','examples'))

import scales

#scale_scenarios = scales.create_scenarios()
#ratio_scenarios = ratios.create_scenarios()
ratios_and_scales_scenarios = ratios_and_scales.create_scenarios()
config_scenarios = different_config_settings.create_scenarios()

default_config = yaml.load(open(join(this_dir, 'default-cfg.yml')))

script = ""
for scen in ratios_and_scales_scenarios:
    print"*" * 160
    print 'Generating scenario:', scen.scenario
    print "*" * 160
    print scen
    scen.generate()
    script+="%s || true\n"%scen.script_path

run_all_script_path = join(scenarios_dir, 'run_all.sh')
run_all_script = open(run_all_script_path, 'wb')
run_all_script.write(script)
chmod(run_all_script_path, 0755)