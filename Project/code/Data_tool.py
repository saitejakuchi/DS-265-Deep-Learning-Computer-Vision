#!/usr/bin/python3
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tool for creating ZIP/PNG based datasets."""

import functools
import gzip
import io
import json
import os
import pickle
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import click
import numpy as np
import PIL.Image
from tqdm import tqdm
from bart.python.bart import bart
import sys, os
sys.path.insert(0, './bart/python')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOOLBOX_PATH"]    = './bart'
sys.path.append('./bart/python')

processed_files = set(['file_brain_AXFLAIR_203_6000942', 'file_brain_AXFLAIR_202_6000531', 'file_brain_AXT1POST_203_6000760', 'file_brain_AXT1POST_205_2050102', 'file_brain_AXT1PRE_200_6002128', 'file_brain_AXT1POST_200_6002392', 'file_brain_AXT2_207_2070422', 'file_brain_AXT1POST_200_6002033', 'file_brain_AXT1POST_205_2050159', 'file_brain_AXT2_200_2000175', 'file_brain_AXT2_200_6001972', 'file_brain_AXT2_204_2040073', 'file_brain_AXT1POST_208_2080154', 'file_brain_AXT2_202_2020290', 'file_brain_AXT1POST_207_2070034', 'file_brain_AXT2_205_2050244', 'file_brain_AXT2_203_2030254', 'file_brain_AXT2_210_6001546', 'file_brain_AXT2_209_2090077', 'file_brain_AXT2_210_2100368', 'file_brain_AXT1_201_6002779', 'file_brain_AXT2_202_2020272', 'file_brain_AXT2_200_6002012', 'file_brain_AXT1_202_2020509', 'file_brain_AXT2_200_2000080', 'file_brain_AXT1POST_202_6000269', 'file_brain_AXT2_204_2040011', 'file_brain_AXFLAIR_202_6000421', 'file_brain_AXT1POST_207_2070357', 'file_brain_AXT1_201_6002836', 'file_brain_AXT1POST_203_6000657', 'file_brain_AXT2_201_2010323', 'file_brain_AXT2_200_6002465', 'file_brain_AXT2_209_6001432', 'file_brain_AXT2_209_2090266', 'file_brain_AXT2_200_6002539', 'file_brain_AXT2_205_6000022', 'file_brain_AXT2_210_2100311', 'file_brain_AXT2_201_2010519', 'file_brain_AXT1_202_2020109', 'file_brain_AXT2_202_2020116', 'file_brain_AXT2_207_2070777', 'file_brain_AXT1PRE_209_6001122', 'file_brain_AXT1POST_201_6002668', 'file_brain_AXT1PRE_210_6001893', 'file_brain_AXT2_210_6001694', 'file_brain_AXT2_208_2080585', 'file_brain_AXFLAIR_209_6001397', 'file_brain_AXT2_201_2010455', 'file_brain_AXT1PRE_203_6000860', 'file_brain_AXT1PRE_206_6000217', 'file_brain_AXT2_210_6001909', 'file_brain_AXT1POST_207_2070133', 'file_brain_AXT1PRE_200_6002047', 'file_brain_AXT2_200_2000360', 'file_brain_AXT2_202_2020331', 'file_brain_AXT2_210_6001651', 'file_brain_AXT1POST_201_6002862', 'file_brain_AXT2_207_2070090', 'file_brain_AXT2_200_6002261', 'file_brain_AXT1POST_208_2080240', 'file_brain_AXFLAIR_202_6000508', 'file_brain_AXT2_200_2000092', 'file_brain_AXT1POST_207_2070786', 'file_brain_AXT2_202_2020365', 'file_brain_AXT1POST_209_6001236', 'file_brain_AXT1_202_2020334', 'file_brain_AXT2_210_6001518', 'file_brain_AXT2_209_6001459', 'file_brain_AXT1_202_6000312', 'file_brain_AXT2_209_6001317', 'file_brain_AXT2_201_2010174', 'file_brain_AXT1POST_201_6002683', 'file_brain_AXFLAIR_200_6002641', 'file_brain_AXT1PRE_206_6000196', 'file_brain_AXT1PRE_203_6000779', 'file_brain_AXT2_210_6001524', 'file_brain_AXT2_200_2000600', 'file_brain_AXT2_206_2060079', 'file_brain_AXT2_207_2070325', 'file_brain_AXT1POST_208_2080421', 'file_brain_AXT2_201_2010201', 'file_brain_AXT2_208_2080303', 'file_brain_AXT1POST_208_2080208', 'file_brain_AXT2_200_2000407', 'file_brain_AXT2_200_2000357', 'file_brain_AXT1POST_208_2080493', 'file_brain_AXT2_205_2050130', 'file_brain_AXT1POST_210_6001790', 'file_brain_AXT2_201_2010241', 'file_brain_AXT2_209_6001436', 'file_brain_AXT2_208_2080635', 'file_brain_AXT1POST_200_6002305', 'file_brain_AXT1POST_207_2070490', 'file_brain_AXT1PRE_201_6002849', 'file_brain_AXT2_205_2050080', 'file_brain_AXT2_208_2080192', 'file_brain_AXT2_201_2010588', 'file_brain_AXT1PRE_203_6000707', 'file_brain_AXT2_200_6002530', 'file_brain_AXT1POST_201_6002673', 'file_brain_AXT2_209_2090406', 'file_brain_AXT1POST_205_2050059', 'file_brain_AXT2_207_2070770', 'file_brain_AXT2_210_6001491', 'file_brain_AXT2_202_2020067', 'file_brain_AXT2_208_2080050', 'file_brain_AXT2_202_2020519', 'file_brain_AXT1POST_207_2070241', 'file_brain_AXT2_201_2010132', 'file_brain_AXT1_202_6000534', 'file_brain_AXT2_206_2060029', 'file_brain_AXT2_200_6002375', 'file_brain_AXT2_207_2070562', 'file_brain_AXT2_202_2020261', 'file_brain_AXT1POST_210_6001672', 'file_brain_AXT2_208_2080722', 'file_brain_AXT2_209_2090333', 'file_brain_AXT2_200_6002151', 'file_brain_AXT2_207_2070039', 'file_brain_AXT1_202_6000339', 'file_brain_AXT2_201_6002681', 'file_brain_AXT2_209_6001384', 'file_brain_AXT1_201_6002717', 'file_brain_AXFLAIR_202_6000539', 'file_brain_AXT2_200_6002228', 'file_brain_AXT1_202_6000305', 'file_brain_AXFLAIR_201_6002960', 'file_brain_AXT2_206_2060027', 'file_brain_AXT2_209_6001025', 'file_brain_AXT2_210_6001617', 'file_brain_AXT2_200_2000621', 'file_brain_AXT1POST_208_2080704', 'file_brain_AXT1POST_207_2070214', 'file_brain_AXT1POST_201_6002817', 'file_brain_AXT2_210_2100365', 'file_brain_AXT2_207_2070834', 'file_brain_AXFLAIR_201_6002941', 'file_brain_AXT2_207_2070500', 'file_brain_AXT2_208_2080186', 'file_brain_AXT1POST_200_6002340', 'file_brain_AXT2_202_2020150', 'file_brain_AXT2_210_6001677', 'file_brain_AXT2_209_2090151', 'file_brain_AXT1POST_201_6002703', 'file_brain_AXFLAIR_201_6002993', 'file_brain_AXT2_207_2070198', 'file_brain_AXT1PRE_203_6000821', 'file_brain_AXT2_210_6001737', 'file_brain_AXT2_207_2070709', 'file_brain_AXT1POST_208_2080352', 'file_brain_AXT1POST_203_6000861', 'file_brain_AXT2_207_2070417', 'file_brain_AXFLAIR_201_6002914', 'file_brain_AXT2_207_2070280', 'file_brain_AXT2_210_2100201', 'file_brain_AXT2_202_2020065', 'file_brain_AXT2_202_2020489', 'file_brain_AXT2_207_2070703', 'file_brain_AXT2_210_2100334', 'file_brain_AXT2_209_2090339', 'file_brain_AXT2_210_2100276', 'file_brain_AXFLAIR_200_6002469', 'file_brain_AXT1POST_202_6000302', 'file_brain_AXT1POST_207_2070099', 'file_brain_AXT2_210_6001944', 'file_brain_AXT2_200_6002532', 'file_brain_AXT2_201_2010332', 'file_brain_AXT2_206_2060013', 'file_brain_AXT2_200_6002545', 'file_brain_AXT1PRE_200_6002309', 'file_brain_AXT1_202_2020478', 'file_brain_AXT1PRE_210_6001656', 'file_brain_AXT2_210_2100136', 'file_brain_AXT2_200_6002308', 'file_brain_AXT2_207_2070100', 'file_brain_AXT2_202_2020016', 'file_brain_AXFLAIR_201_6003003', 'file_brain_AXT2_200_6002551', 'file_brain_AXT1POST_209_6001250', 'file_brain_AXT2_210_2100105', 'file_brain_AXT2_210_2100343', 'file_brain_AXT2_202_2020500', 'file_brain_AXT1_202_2020186', 'file_brain_AXT2_209_2090215', 'file_brain_AXT2_203_2030092', 'file_brain_AXT2_200_6002116', 'file_brain_AXT2_202_2020039', 'file_brain_AXT2_200_2000290', 'file_brain_AXT2_201_2010029', 'file_brain_AXT2_202_2020555', 'file_brain_AXT1POST_207_2070251', 'file_brain_AXT2_208_2080611', 'file_brain_AXT1PRE_210_6001891', 'file_brain_AXT1PRE_203_6000795', 'file_brain_AXT2_209_6001328', 'file_brain_AXT2_208_2080277', 'file_brain_AXT2_207_2070822', 'file_brain_AXT1_202_2020496', 'file_brain_AXFLAIR_201_6002972', 'file_brain_AXT2_210_2100260', 'file_brain_AXT2_200_6002446', 'file_brain_AXT2_207_2070249', 'file_brain_AXT1POST_203_6000774', 'file_brain_AXT1POST_200_6002077', 'file_brain_AXT2_208_2080013', 'file_brain_AXT2_200_2000020', 'file_brain_AXT2_208_2080091', 'file_brain_AXT1PRE_200_6002247', 'file_brain_AXT2_209_6001261', 'file_brain_AXT1_202_2020209', 'file_brain_AXT2_201_2010168', 'file_brain_AXT1_202_2020377', 'file_brain_AXT1POST_207_2070179', 'file_brain_AXT2_210_2100086', 'file_brain_AXT2_200_6002262', 'file_brain_AXT2_207_2070165', 'file_brain_AXT2_202_2020022', 'file_brain_AXT2_207_2070458', 'file_brain_AXT2_210_6001583', 'file_brain_AXT1POST_208_2080127', 'file_brain_AXT2_210_6001506', 'file_brain_AXT2_210_6001711', 'file_brain_AXT2_201_2010398', 'file_brain_AXT2_200_2000488', 'file_brain_AXT2_209_6001420', 'file_brain_AXT1POST_208_2080637', 'file_brain_AXT2_203_2030135', 'file_brain_AXT1_202_2020570', 'file_brain_AXT2_200_6002387', 'file_brain_AXT1POST_208_2080664', 'file_brain_AXT1POST_209_6001434', 'file_brain_AXT2_207_2070608', 'file_brain_AXT2_210_6001709', 'file_brain_AXT1POST_200_6002229', 'file_brain_AXT2_203_2030096', 'file_brain_AXT2_201_2010239', 'file_brain_AXFLAIR_200_6002629', 'file_brain_AXT1POST_203_6000776', 'file_brain_AXT1POST_200_6002122', 'file_brain_AXT2_208_2080526', 'file_brain_AXFLAIR_210_6001914', 'file_brain_AXT1POST_203_6000867', 'file_brain_AXT1POST_202_6000488', 'file_brain_AXT1POST_210_6001850', 'file_brain_AXT2_202_2020127', 'file_brain_AXT1POST_202_6000588', 'file_brain_AXT2_201_2010416', 'file_brain_AXT1POST_205_2050026', 'file_brain_AXT2_200_2000486', 'file_brain_AXT2_210_6001941', 'file_brain_AXT2_210_6001763', 'file_brain_AXT2_202_2020214', 'file_brain_AXT2_210_6001747', 'file_brain_AXT2_207_2070460', 'file_brain_AXT1_206_6000204', 'file_brain_AXT2_208_2080051', 'file_brain_AXT2_208_2080126', 'file_brain_AXT2_200_6002500', 'file_brain_AXT2_210_6001534', 'file_brain_AXT1PRE_203_6000837', 'file_brain_AXT1_202_6000347', 'file_brain_AXT2_201_2010571', 'file_brain_AXT1POST_207_2070143', 'file_brain_AXT1POST_203_6000814', 'file_brain_AXT1POST_201_6002696', 'file_brain_AXT2_203_2030256', 'file_brain_AXT2_207_2070208', 'file_brain_AXT2_205_2050062', 'file_brain_AXT2_208_2080445', 'file_brain_AXT1PRE_200_6002304', 'file_brain_AXFLAIR_201_6002940', 'file_brain_AXT2_210_2100090', 'file_brain_AXT2_202_2020273', 'file_brain_AXT2_203_2030352', 'file_brain_AXT1PRE_203_6000672', 'file_brain_AXT2_204_2040042', 'file_brain_AXT1PRE_200_6002408', 'file_brain_AXT2_200_6002381', 'file_brain_AXT2_208_2080333', 'file_brain_AXT1POST_201_6002737', 'file_brain_AXFLAIR_201_6002876', 'file_brain_AXFLAIR_201_6002888', 'file_brain_AXT2_200_6001980', 'file_brain_AXT1POST_203_6000628', 'file_brain_AXT2_200_6002528', 'file_brain_AXT2_200_2000592', 'file_brain_AXT1POST_201_6002774', 'file_brain_AXT2_201_2010156', 'file_brain_AXFLAIR_200_6002452', 'file_brain_AXT2_200_6002655', 'file_brain_AXT2_201_2010585', 'file_brain_AXT2_209_6000981', 'file_brain_AXT2_207_2070054', 'file_brain_AXT2_202_2020297', 'file_brain_AXT2_200_2000362', 'file_brain_AXT2_201_2010626', 'file_brain_AXT1POST_208_2080626', 'file_brain_AXFLAIR_200_6002467', 'file_brain_AXT2_210_6001928', 'file_brain_AXT1_202_2020098', 'file_brain_AXT2_208_2080329', 'file_brain_AXT2_207_2070194', 'file_brain_AXT1POST_207_2070505', 'file_brain_AXT1POST_207_2070404', 'file_brain_AXT2_207_2070145', 'file_brain_AXT2_201_2010628', 'file_brain_AXT2_208_2080713', 'file_brain_AXT2_210_2100306', 'file_brain_AXT1POST_209_6001199', 'file_brain_AXT1POST_207_2070612', 'file_brain_AXT2_210_2100345', 'file_brain_AXT2_209_6001248', 'file_brain_AXT2_201_2010625', 'file_brain_AXT1POST_205_6000162'] + ['file_brain_AXFLAIR_210_6001521', 'file_brain_AXT2_202_2020168', 'file_brain_AXT1POST_200_6002089', 'file_brain_AXT1PRE_200_6002041', 'file_brain_AXT2_200_2000321', 'file_brain_AXT2_200_2000173', 'file_brain_AXT2_202_2020162', 'file_brain_AXT1_202_2020468', 'file_brain_AXT2_202_2020562', 'file_brain_AXT1PRE_205_6000178', 'file_brain_AXT1POST_203_6000597', 'file_brain_AXT1POST_200_6001959', 'file_brain_AXT1_202_2020076', 'file_brain_AXT2_210_2100359', 'file_brain_AXT1PRE_203_6000614', 'file_brain_AXT1PRE_209_6001470', 'file_brain_AXT2_201_2010374', 'file_brain_AXT1POST_208_2080392', 'file_brain_AXT2_210_2100165', 'file_brain_AXT2_205_2050158', 'file_brain_AXT1POST_200_6002065', 'file_brain_AXT2_200_2000365', 'file_brain_AXT1POST_208_2120069', 'file_brain_AXT2_201_2010121', 'file_brain_AXT2_200_2000030', 'file_brain_AXT2_206_2060089', 'file_brain_AXT1POST_200_6002003', 'file_brain_AXT1PRE_203_6000699', 'file_brain_AXT1PRE_201_6002691', 'file_brain_AXT2_200_2000229', 'file_brain_AXT2_206_2060088', 'file_brain_AXT2_206_6000242', 'file_brain_AXT2_200_6001958', 'file_brain_AXT2_207_2070731', 'file_brain_AXT2_209_2090228', 'file_brain_AXT2_210_6001873', 'file_brain_AXT1PRE_203_6000805', 'file_brain_AXT2_202_6000311', 'file_brain_AXT1_202_6000408', 'file_brain_AXT2_201_2010095', 'file_brain_AXT2_200_6001968', 'file_brain_AXT1POST_205_2050146', 'file_brain_AXT2_200_2000372', 'file_brain_AXT1_202_2020199', 'file_brain_AXT2_201_2010381', 'file_brain_AXT2_210_6001681', 'file_brain_AXT1POST_207_2070846', 'file_brain_AXT2_201_6002742', 'file_brain_AXT2_200_2000301', 'file_brain_AXT2_207_2070415', 'file_brain_AXT2_203_2030004', 'file_brain_AXT2_206_6000257', 'file_brain_AXT2_200_2000224', 'file_brain_AXT1POST_201_6002731', 'file_brain_AXT1PRE_203_6000862', 'file_brain_AXT1POST_209_6001064', 'file_brain_AXT2_209_6001453', 'file_brain_AXT2_203_2030397', 'file_brain_AXT1POST_207_2070242', 'file_brain_AXT1POST_206_6000235', 'file_brain_AXT1POST_208_2080553', 'file_brain_AXT1POST_209_6001234', 'file_brain_AXT1POST_201_6002707', 'file_brain_AXT1_202_2020009', 'file_brain_AXFLAIR_210_6001499', 'file_brain_AXT2_200_6002561', 'file_brain_AXT1_202_2020551', 'file_brain_AXT1POST_208_2080275', 'file_brain_AXT1POST_201_6002749', 'file_brain_AXT1POST_200_6002349', 'file_brain_AXT2_209_6001378', 'file_brain_AXT1POST_208_2080304', 'file_brain_AXT2_202_2020101', 'file_brain_AXT1POST_208_2080227', 'file_brain_AXT1_202_2020323', 'file_brain_AXT2_200_6002160', 'file_brain_AXT2_208_2080098', 'file_brain_AXT2_202_2020469', 'file_brain_AXT2_209_2090117', 'file_brain_AXT2_204_2040019', 'file_brain_AXFLAIR_202_6000511', 'file_brain_AXT2_206_6000186', 'file_brain_AXT1POST_207_2070365', 'file_brain_AXT1POST_202_6000281', 'file_brain_AXT1POST_208_2080383', 'file_brain_AXT1POST_207_2070327', 'file_brain_AXT1POST_205_6000148', 'file_brain_AXT1POST_207_2070147', 'file_brain_AXT1PRE_203_6000683', 'file_brain_AXT2_209_2090054', 'file_brain_AXT2_201_2010023', 'file_brain_AXT1PRE_209_6001178', 'file_brain_AXT2_200_6002034', 'file_brain_AXT2_205_6000097', 'file_brain_AXT1_202_2020511', 'file_brain_AXT2_205_2050060', 'file_brain_AXT1PRE_209_6001235', 'file_brain_AXFLAIR_203_6000898', 'file_brain_AXT2_207_2070734', 'file_brain_AXT2_210_6001755', 'file_brain_AXT1PRE_203_6000651', 'file_brain_AXT2_200_2000165', 'file_brain_AXT2_209_6001396', 'file_brain_AXT2_207_2070372', 'file_brain_AXFLAIR_209_6001377', 'file_brain_AXT2_207_2070677', 'file_brain_AXT2_207_2070467', 'file_brain_AXFLAIR_210_6001513', 'file_brain_AXT1POST_207_2070479', 'file_brain_AXT1_202_2020301', 'file_brain_AXT2_200_6001985', 'file_brain_AXT2_210_2100026', 'file_brain_AXT1POST_207_2070083', 'file_brain_AXT1PRE_200_6002371', 'file_brain_AXT2_201_2010427', 'file_brain_AXT2_203_2030219', 'file_brain_AXT2_207_2070207', 'file_brain_AXT1POST_201_6002760', 'file_brain_AXT2_208_2080408', 'file_brain_AXT2_205_2050165', 'file_brain_AXT1PRE_201_6002755', 'file_brain_AXT1POST_200_6001996', 'file_brain_AXT2_205_2050212', 'file_brain_AXT2_205_6000182', 'file_brain_AXT2_209_6001304', 'file_brain_AXT2_200_6001994', 'file_brain_AXT2_208_2080519', 'file_brain_AXT1POST_208_2080504', 'file_brain_AXT2_203_2030139', 'file_brain_AXT2_209_6001030', 'file_brain_AXT2_210_2100107', 'file_brain_AXT2_205_2050262', 'file_brain_AXT2_202_2020424', 'file_brain_AXT1POST_207_2070651', 'file_brain_AXT2_201_2010039', 'file_brain_AXT2_202_2020407', 'file_brain_AXT1POST_210_6001771', 'file_brain_AXT2_201_2010262', 'file_brain_AXT1POST_208_2080634', 'file_brain_AXT2_210_6001933', 'file_brain_AXT2_207_2070514', 'file_brain_AXT2_201_2010125', 'file_brain_AXT1PRE_209_6001324', 'file_brain_AXFLAIR_201_6003024', 'file_brain_AXT1POST_209_6001279', 'file_brain_AXT2_209_2090220', 'file_brain_AXT1PRE_210_6001684', 'file_brain_AXT2_210_2100216', 'file_brain_AXT2_200_6002515', 'file_brain_AXT1_206_2060051', 'file_brain_AXT2_209_2090102', 'file_brain_AXFLAIR_202_6000483', 'file_brain_AXT1POST_207_2070116', 'file_brain_AXT2_200_6002092', 'file_brain_AXT2_206_6000207', 'file_brain_AXT2_210_6001588', 'file_brain_AXT2_207_2070360', 'file_brain_AXT1_202_2020068', 'file_brain_AXT2_200_2000319', 'file_brain_AXT2_205_6000019', 'file_brain_AXT2_202_2020174', 'file_brain_AXT2_205_6000049', 'file_brain_AXT2_203_2030203', 'file_brain_AXT2_210_6001598', 'file_brain_AXT2_205_6000139', 'file_brain_AXT1POST_202_6000341', 'file_brain_AXT2_200_6002569', 'file_brain_AXT1PRE_203_6000705', 'file_brain_AXT2_202_2020275', 'file_brain_AXT2_208_2080361', 'file_brain_AXT2_204_2040063', 'file_brain_AXT2_200_6002455', 'file_brain_AXT1POST_200_6002046', 'file_brain_AXFLAIR_205_6000181', 'file_brain_AXT2_209_6001108', 'file_brain_AXT2_200_2000220', 'file_brain_AXT1_201_6002811', 'file_brain_AXT2_206_2060033', 'file_brain_AXT1POST_208_2080075', 'file_brain_AXT2_201_2010048', 'file_brain_AXT1POST_201_6002772', 'file_brain_AXT2_200_2000088', 'file_brain_AXT2_208_2080703', 'file_brain_AXT1PRE_201_6002904', 'file_brain_AXT1PRE_203_6000773', 'file_brain_AXT1POST_205_6000035', 'file_brain_AXT2_200_6002476', 'file_brain_AXT2_208_2080340', 'file_brain_AXT1POST_208_2080082', 'file_brain_AXT2_207_2070195', 'file_brain_AXT2_202_2020483', 'file_brain_AXFLAIR_210_6001901', 'file_brain_AXT2_200_6002534', 'file_brain_AXT2_204_2040057', 'file_brain_AXT2_208_2080106', 'file_brain_AXFLAIR_201_6002878', 'file_brain_AXT2_208_2080223', 'file_brain_AXT2_200_6001971', 'file_brain_AXT2_202_2020229', 'file_brain_AXT1POST_200_6002337', 'file_brain_AXT1_206_2060097', 'file_brain_AXT1PRE_203_6000627', 'file_brain_AXT2_205_2050261', 'file_brain_AXT2_202_2020299', 'file_brain_AXT2_209_6001476', 'file_brain_AXT2_207_2070392', 'file_brain_AXT1POST_205_6000180', 'file_brain_AXT2_200_6002280', 'file_brain_AXT2_209_6001175', 'file_brain_AXT2_200_2000214', 'file_brain_AXT2_208_2080148', 'file_brain_AXT2_201_2010543', 'file_brain_AXT2_208_2080600', 'file_brain_AXT2_209_2090210', 'file_brain_AXT2_207_2070204', 'file_brain_AXT2_201_2010215', 'file_brain_AXT1POST_208_2080380', 'file_brain_AXFLAIR_202_6000543', 'file_brain_AXT1POST_208_2080062', 'file_brain_AXT2_201_2010303', 'file_brain_AXT1POST_208_2080213', 'file_brain_AXT2_210_6001859', 'file_brain_AXT1POST_203_6000869', 'file_brain_AXT1_202_2020195', 'file_brain_AXT2_203_2030339', 'file_brain_AXT2_209_6001168', 'file_brain_AXT2_207_2070653', 'file_brain_AXT2_200_2000383', 'file_brain_AXT1POST_210_6001787', 'file_brain_AXT2_204_2040055', 'file_brain_AXT1PRE_200_6002251', 'file_brain_AXT1PRE_200_6002358', 'file_brain_AXT1PRE_203_6000894', 'file_brain_AXT1POST_207_2070123', 'file_brain_AXT1PRE_203_6000785', 'file_brain_AXFLAIR_200_6002543', 'file_brain_AXT2_201_2010524', 'file_brain_AXT2_207_2070078', 'file_brain_AXT1POST_200_6002396', 'file_brain_AXT2_206_6000198', 'file_brain_AXT1PRE_203_6000643', 'file_brain_AXT1POST_201_6002795', 'file_brain_AXT2_206_6000261', 'file_brain_AXT1PRE_200_6002072', 'file_brain_AXT2_202_2020288', 'file_brain_AXT1POST_202_6000417', 'file_brain_AXT2_202_2020534', 'file_brain_AXT2_201_2010292', 'file_brain_AXT1POST_200_6002227', 'file_brain_AXT1POST_210_6001735', 'file_brain_AXT2_201_6002961', 'file_brain_AXT1_202_2020134', 'file_brain_AXT1POST_203_6000827', 'file_brain_AXT1POST_205_2050259', 'file_brain_AXT1POST_201_6002735', 'file_brain_AXT1POST_207_2070804', 'file_brain_AXT1POST_202_6000580', 'file_brain_AXT2_209_6001077', 'file_brain_AXT1POST_208_2080469', 'file_brain_AXT2_200_6002383', 'file_brain_AXT1PRE_210_6001827', 'file_brain_AXT1_202_2020418', 'file_brain_AXFLAIR_203_6000875', 'file_brain_AXT2_204_2040070', 'file_brain_AXT1POST_207_2070354', 'file_brain_AXT1POST_205_2050226', 'file_brain_AXT1POST_207_2070418', 'file_brain_AXT2_209_6000983', 'file_brain_AXT1POST_201_6002920', 'file_brain_AXT2_208_2080040', 'file_brain_AXT2_209_2090226', 'file_brain_AXT2_209_2090305', 'file_brain_AXT2_209_2090222', 'file_brain_AXT1POST_203_6000661', 'file_brain_AXT1POST_207_2070809', 'file_brain_AXT2_203_2030133', 'file_brain_AXT1POST_209_6001264', 'file_brain_AXT1POST_203_6000877', 'file_brain_AXT1POST_208_2080345', 'file_brain_AXT2_209_2090029', 'file_brain_AXT2_200_6002218', 'file_brain_AXT1POST_210_6001797', 'file_brain_AXT2_200_2000576', 'file_brain_AXT1POST_209_6001109', 'file_brain_AXT2_202_2020401', 'file_brain_AXT2_210_6001830', 'file_brain_AXT1PRE_200_6002243', 'file_brain_AXT2_205_2050121', 'file_brain_AXT2_208_2080160', 'file_brain_AXT1PRE_205_6000096', 'file_brain_AXT2_208_2080439', 'file_brain_AXT1POST_210_6001699', 'file_brain_AXT2_208_2080503', 'file_brain_AXT2_208_2080401', 'file_brain_AXT2_203_2030086', 'file_brain_AXT2_200_6002021', 'file_brain_AXT2_208_2080552', 'file_brain_AXT2_210_6001663', 'file_brain_AXT2_208_2080395', 'file_brain_AXT2_205_6000167', 'file_brain_AXT1POST_203_6000848', 'file_brain_AXT2_210_2100007', 'file_brain_AXT1PRE_203_6000728', 'file_brain_AXT2_202_2020329', 'file_brain_AXT2_201_2010266', 'file_brain_AXT1_202_2020156', 'file_brain_AXT1POST_207_2070450', 'file_brain_AXT2_200_2000522', 'file_brain_AXT1POST_203_6000644', 'file_brain_AXT2_210_2100346', 'file_brain_AXT2_205_2050061', 'file_brain_AXT2_209_2090133', 'file_brain_AXT2_205_2050012', 'file_brain_AXT1POST_201_6002814', 'file_brain_AXT2_209_6001437', 'file_brain_AXT1PRE_210_6001710', 'file_brain_AXT2_208_2080561', 'file_brain_AXT1PRE_209_6001285', 'file_brain_AXT1POST_208_2080624', 'file_brain_AXT2_208_2080398', 'file_brain_AXT2_200_2000329', 'file_brain_AXT2_200_2000355', 'file_brain_AXT2_200_6002062', 'file_brain_AXFLAIR_210_6001953', 'file_brain_AXT1POST_207_2070095', 'file_brain_AXT2_205_2050225', 'file_brain_AXT1POST_209_6001417', 'file_brain_AXT2_203_2030333', 'file_brain_AXT1POST_200_6002259', 'file_brain_AXT2_210_6001913', 'file_brain_AXT1POST_205_2050189', 'file_brain_AXFLAIR_203_6000912', 'file_brain_AXT2_207_2070226', 'file_brain_AXT1PRE_206_6000201', 'file_brain_AXT2_203_2030335', 'file_brain_AXT2_208_2080170', 'file_brain_AXT1PRE_210_6001844', 'file_brain_AXT1_202_2020167', 'file_brain_AXT1POST_201_6002778', 'file_brain_AXT2_201_2010430', 'file_brain_AXT1POST_207_2070010', 'file_brain_AXT1POST_207_2070553', 'file_brain_AXT2_210_2100087', 'file_brain_AXT2_201_2010514', 'file_brain_AXT2_205_2050185', 'file_brain_AXT2_200_2000071', 'file_brain_AXT2_208_2080037', 'file_brain_AXT1POST_200_6002413', 'file_brain_AXT1POST_201_6002680', 'file_brain_AXT1PRE_209_6000986', 'file_brain_AXT2_203_2030341', 'file_brain_AXFLAIR_200_6002442', 'file_brain_AXT2_200_6002630', 'file_brain_AXT1POST_209_6001320', 'file_brain_AXT2_203_2030199', 'file_brain_AXT2_200_6002272', 'file_brain_AXT2_209_2090394', 'file_brain_AXT1POST_207_2070127', 'file_brain_AXT2_203_2030280', 'file_brain_AXT2_200_6002626', 'file_brain_AXT2_207_2070656', 'file_brain_AXT2_207_2070408', 'file_brain_AXT2_201_2010072', 'file_brain_AXT2_208_2080029', 'file_brain_AXT2_210_2100066', 'file_brain_AXT2_208_2080509', 'file_brain_AXT2_208_2080581', 'file_brain_AXT2_207_2070380', 'file_brain_AXT1PRE_205_6000063', 'file_brain_AXT2_208_2080364', 'file_brain_AXT1POST_207_2070259', 'file_brain_AXT2_206_2060040', 'file_brain_AXT2_203_2030302', 'file_brain_AXT1_202_2020248', 'file_brain_AXFLAIR_202_6000557', 'file_brain_AXT1_202_2020184', 'file_brain_AXT1POST_208_2080244', 'file_brain_AXT1POST_209_6001458', 'file_brain_AXFLAIR_202_6000474', 'file_brain_AXT2_207_2070712', 'file_brain_AXT1PRE_203_6000620', 'file_brain_AXT2_201_2010384', 'file_brain_AXT2_203_2030201', 'file_brain_AXT1POST_207_2070009', 'file_brain_AXT2_201_2010413', 'file_brain_AXT2_201_2010151', 'file_brain_AXT2_209_2090109', 'file_brain_AXT1POST_208_2080422', 'file_brain_AXT1POST_208_2080477', 'file_brain_AXT2_207_2070062', 'file_brain_AXT1POST_207_2070765', 'file_brain_AXT2_209_6001427', 'file_brain_AXT2_209_2090074', 'file_brain_AXT1POST_205_2050148', 'file_brain_AXT2_210_6001558', 'file_brain_AXFLAIR_200_6002595', 'file_brain_AXT1POST_202_6000395', 'file_brain_AXT1POST_200_6002188', 'file_brain_AXT1POST_207_2070547', 'file_brain_AXT1PRE_203_6000791', 'file_brain_AXT2_200_2000441', 'file_brain_AXT2_209_2090290', 'file_brain_AXT2_202_2020148', 'file_brain_AXT2_200_2000642', 'file_brain_AXT2_200_2000123', 'file_brain_AXT2_209_2090182', 'file_brain_AXT2_202_2020491', 'file_brain_AXT1PRE_203_6000873', 'file_brain_AXT2_210_2100114', 'file_brain_AXT2_202_2020333', 'file_brain_AXT2_209_2090234', 'file_brain_AXT2_209_2090104', 'file_brain_AXT2_210_2100127', 'file_brain_AXT2_205_2050053', 'file_brain_AXT2_200_6002314', 'file_brain_AXT1POST_203_6000712', 'file_brain_AXT2_210_2100293', 'file_brain_AXT2_200_2000137', 'file_brain_AXT1POST_210_6001645', 'file_brain_AXFLAIR_201_6003018', 'file_brain_AXT2_202_2020075', 'file_brain_AXT1POST_208_2080393', 'file_brain_AXT2_210_6001783', 'file_brain_AXT1POST_205_2050034', 'file_brain_AXT2_200_2000187', 'file_brain_AXT2_207_2070192', 'file_brain_AXT2_203_2030182', 'file_brain_AXT1PRE_200_6002032', 'file_brain_AXT2_209_2090024', 'file_brain_AXT2_208_2080529', 'file_brain_AXT2_202_2020208', 'file_brain_AXT2_209_6001321', 'file_brain_AXT2_208_2080222', 'file_brain_AXT2_204_2040083', 'file_brain_AXT2_208_2080587', 'file_brain_AXT2_202_2020004', 'file_brain_AXT1PRE_210_6001848', 'file_brain_AXT2_209_2090298', 'file_brain_AXT2_210_2100322', 'file_brain_AXT1POST_207_2070635', 'file_brain_AXT2_205_2050115', 'file_brain_AXT1PRE_209_6000978', 'file_brain_AXT2_210_2100080', 'file_brain_AXT2_200_2000393', 'file_brain_AXT2_207_2070142', 'file_brain_AXT1PRE_203_6000801'])

# ----------------------------------------------------------------------------
# Parse a 'M,N' or 'MxN' integer tuple.
# Example: '4x2' returns (4,2)

def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise click.ClickException(f'cannot parse tuple {s}')

# ----------------------------------------------------------------------------


def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

# ----------------------------------------------------------------------------


def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

# ----------------------------------------------------------------------------


def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION

# ----------------------------------------------------------------------------


def open_image_folder(source_dir, *, max_images: Optional[int]):
    input_images = [str(f) for f in sorted(Path(source_dir).rglob(
        '*')) if is_image_ext(f) and os.path.isfile(f)]
    arch_fnames = {fname: os.path.relpath(fname, source_dir).replace(
        '\\', '/') for fname in input_images}
    max_idx = maybe_min(len(input_images), max_images)

    # Load labels.
    labels = dict()
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            data = json.load(file)['labels']
            if data is not None:
                labels = {x[0]: x[1] for x in data}

    # No labels available => determine from top-level directory names.
    if len(labels) == 0:
        toplevel_names = {arch_fname: arch_fname.split(
            '/')[0] if '/' in arch_fname else '' for arch_fname in arch_fnames.values()}
        toplevel_indices = {toplevel_name: idx for idx, toplevel_name in enumerate(
            sorted(set(toplevel_names.values())))}
        if len(toplevel_indices) > 1:
            labels = {arch_fname: toplevel_indices[toplevel_name]
                      for arch_fname, toplevel_name in toplevel_names.items()}

    def iterate_images():
        for idx, fname in enumerate(input_images):
            img = np.array(PIL.Image.open(fname))
            yield dict(img=img, label=labels.get(arch_fnames.get(fname)))
            if idx >= max_idx - 1:
                break
    return max_idx, iterate_images()

# ----------------------------------------------------------------------------


def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f)
                        for f in sorted(z.namelist()) if is_image_ext(f)]
        max_idx = maybe_min(len(input_images), max_images)

        # Load labels.
        labels = dict()
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                data = json.load(file)['labels']
                if data is not None:
                    labels = {x[0]: x[1] for x in data}

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = np.array(PIL.Image.open(file))
                yield dict(img=img, label=labels.get(fname))
                if idx >= max_idx - 1:
                    break
    return max_idx, iterate_images()

# ----------------------------------------------------------------------------


def open_lmdb(lmdb_dir: str, *, max_images: Optional[int]):
    # pyright: ignore [reportMissingImports] # pip install opencv-python
    import cv2
    import lmdb  # pyright: ignore [reportMissingImports] # pip install lmdb

    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        max_idx = maybe_min(txn.stat()['entries'], max_images)

    def iterate_images():
        with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.frombuffer(
                            value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1]  # BGR => RGB
                    except IOError:
                        img = np.array(PIL.Image.open(io.BytesIO(value)))
                    yield dict(img=img, label=None)
                    if idx >= max_idx - 1:
                        break
                except:
                    print(sys.exc_info()[1])

    return max_idx, iterate_images()

# ----------------------------------------------------------------------------


def open_cifar10(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:
        for batch in range(1, 6):
            member = tar.getmember(f'cifar-10-batches-py/data_batch_{batch}')
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')
            images.append(data['data'].reshape(-1, 3, 32, 32))
            labels.append(data['labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1])  # NCHW -> NHWC
    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx - 1:
                break

    return max_idx, iterate_images()

# ----------------------------------------------------------------------------


def open_mnist(images_gz: str, *, max_images: Optional[int]):
    labels_gz = images_gz.replace(
        '-images-idx3-ubyte.gz', '-labels-idx1-ubyte.gz')
    assert labels_gz != images_gz
    images = []
    labels = []

    with gzip.open(images_gz, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    with gzip.open(labels_gz, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0, 0), (2, 2), (2, 2)],
                    'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx - 1:
                break

    return max_idx, iterate_images()

# ----------------------------------------------------------------------------

def save_image_to_disk(data_to_save, id_value, destination_path, save_function, transform_function, image_attr_data, save_to_npy = False):
    idx_str = f'{id_value:08d}'
    # Apply crop and resize.
    if save_to_npy:
        archive_fname = f'{idx_str[:5]}/img{idx_str}.npy'
        cur_image_attrs = {
            'width': data_to_save.shape[2], 'height': data_to_save.shape[1], 'channels': data_to_save.shape[0]}
        path_data = os.path.join(destination_path, archive_fname)
        os.makedirs(os.path.dirname(path_data), exist_ok=True)
        np.save(path_data, data_to_save)
    else:
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
        image_data = transform_function(data_to_save)
        if image_data is None:
            return
        channels = image_data.shape[2] if image_data.ndim == 3 else 1
        cur_image_attrs = {
            'width': image_data.shape[1], 'height': image_data.shape[0], 'channels': channels}
        if image_attr_data is None:
            image_attr_data = cur_image_attrs
            width = image_attr_data['width']
            height = image_attr_data['height']
            if width != height:
                raise click.ClickException(
                    f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if image_attr_data['channels'] not in [1, 3]:
                raise click.ClickException(
                    'Input images must be stored as RGB or grayscale')
            # if width != 2 ** int(np.floor(np.log2(width))):
            #     raise click.ClickException(
            #         'Image width/height after scale and crop are required to be power-of-two')
        elif image_attr_data != cur_image_attrs:
            err = [
                f'  dataset {k}/cur image {k}: {image_attr_data[k]}/{cur_image_attrs[k]}' for k in image_attr_data.keys()]
            raise click.ClickException(
                f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        image_data = PIL.Image.fromarray(image_data, {1: 'L', 3: 'RGB'}[channels])
        image_bits = io.BytesIO()
        image_data.save(image_bits, format='png', compress_level=0, optimize=False)
        save_function(os.path.join(destination_path, archive_fname),
                    image_bits.getbuffer())
    
    return image_attr_data, archive_fname


def open_mridata(src_dir: str, transform_function, destination_path, save_function, close_destination, end_to_end: bool, max_images: Optional[int], use_np = True):
    import xml.etree.ElementTree as etree
    from pathlib import Path
    import re
    import h5py
    from tqdm import tqdm
    import torch
    import fastmri
    from fastmri.data import transforms
    from fastmri.data.mri_data import et_query
    from sigpy.mri.app import EspiritCalib
    import cv2
    from dnnlib.util import Logger
    import time
    
    labels = []
    global processed_files

    if end_to_end:
        dataset_attrs = None
        current_image_count = 0 # 5076 # 272
    
    else:
        images = []
    
    label_ids = {'FLAIR': 0, 'T1PRE': 1, 'T1POST': 2, 'T1': 3, 'T2': 4}
    label_to_id = lambda label_value: label_ids[label_value]
    label_regex_pattern = ".*_AX([a-zA-Z\d]*)_"
    files_data = list(Path(src_dir).glob("*.h5"))
    logger = Logger(f'dataset_logdata2.txt', 'a')
    try:
        # for index, fname in tqdm(enumerate(files_data), total=len(files_data)):
        for index in tqdm(range(0, len(files_data))):
            fname = files_data[index]
            with h5py.File(fname, "r") as hf:
                # mtime = os.path.getctime(fname)
                # mstr = time.ctime(mtime)
                label_data = fname.stem
                if label_data in processed_files:
                    continue
                logger.write(f'\ncurrently in file {label_data}\n')
                matches = re.findall(label_regex_pattern, label_data, re.MULTILINE)
                label_value = matches[0]
                et_root = etree.fromstring(hf["ismrmrd_header"][()])

                enc = ["encoding", "reconSpace", "matrixSize"]
                crop_size = (
                    int(et_query(et_root, enc + ["x"])),
                    int(et_query(et_root, enc + ["y"])),
                )
                if crop_size[1] != crop_size[0]:
                    logger.write(f'{fname} has rectange reconshape {crop_size}. Approx 16 images missed.\n')
                    continue

                masked_kspace = transforms.to_tensor(hf["kspace"][()])
                kspace = np.array( hf['kspace'] )
                if crop_size[0] > masked_kspace.shape[2]:
                    logger.write(f'Skipping {fname} as height of recon {crop_size[0]} is greater than the actual image ksapce size {masked_kspace.shape[2]}.\n')
                    continue
                
                if crop_size[1] > masked_kspace.shape[3]:
                    logger.write(f'Skipping {fname} as width of recon {crop_size[1]} is greater than the actual image ksapce size {masked_kspace.shape[3]}.\n')
                    continue
                
                # NMHWC, N -> batch size, M -> num of scans, C -> real and complex channels
                image = fastmri.ifft2c(masked_kspace)
                batch_count = image.shape[0]
                logger.write(f'Found {batch_count=}\n')
                if use_np:
                    for index in range(batch_count):
                        # kspace_data = masked_kspace[index][:, :, :, 0] + 1j * masked_kspace[index][:, :, :, 1]
                        image_space_data = image[index][:, :, :, 0] + 1j * image[index][:, :, :, 1]
                        gt_ksp = kspace[index]
                        s_maps_ind = bart(1, 'ecalib -m1 -W -c0', gt_ksp.transpose((1, 2, 0))[None,...]).transpose( (3, 1, 2, 0)).squeeze()
                        s_maps_ind_conj = torch.conj(torch.from_numpy(s_maps_ind))
                        # kspace_conj_maps_data = torch.conj(torch.from_numpy(EspiritCalib(kspace_data.numpy(), show_pbar = False).run()))
                        image_data = torch.sum(torch.multiply(image_space_data, s_maps_ind_conj), axis=0)
                        max_value = torch.max(torch.abs(image_data))
                        
                        normalized_data = image_data/max_value
                        cropped_img_data = transforms.center_crop(normalized_data, crop_size)
                        flipped_img_data = np.flip(cropped_img_data.numpy(), axis=0)
                        final_numpy_data = np.array([flipped_img_data.real, flipped_img_data.imag])
                        final_numpy_data = cv2.resize(final_numpy_data[0], (256, 256), interpolation=cv2.INTER_LINEAR) + 1j * cv2.resize(final_numpy_data[1], (256, 256), interpolation=cv2.INTER_LINEAR)
                        final_numpy_data = np.array([final_numpy_data.real, final_numpy_data.imag])
                        dataset_attrs, saved_filename = save_image_to_disk(final_numpy_data, current_image_count, destination_path, save_function, transform_function, dataset_attrs, True)
                        labels.append((saved_filename, label_ids[label_value]))
                        # final_numpy_data_viz = np.array(255*np.abs(flipped_img_data)).astype(np.uint8)
                        # viz_filename = f"{saved_filename.split('/')[-1].split('.')[0]}.png"
                        # cv2.imwrite(viz_filename, final_numpy_data_viz)
                        current_image_count += 1
                    logger.write(f'Finished writing batch images data, {current_image_count=}\n')
                else:                
                    # crop input image
                    batch_images_data = transforms.complex_center_crop(image, crop_size)

                    # absolute value. # NMHW
                    batch_images_data = fastmri.complex_abs(batch_images_data)
                    batch_images_data = fastmri.rss(batch_images_data, dim=1)
                    if end_to_end:
                        for index in range(batch_count):          
                            dataset_attrs, saved_filename = save_image_to_disk(batch_images_data[index], current_image_count, destination_path, save_function, transform_function, dataset_attrs)
                            labels.append((saved_filename, label_ids[label_value]))
                        current_image_count += 1
                    else:
                        images += batch_images_data
                        labels += [label_value] * batch_count

        if not end_to_end:
            labels = list(map(label_to_id, labels))

        max_idx = maybe_min(len(images), max_images)

        def iterate_images():
            for idx, img in enumerate(images):
                yield dict(img=img, label=int(labels[idx]))
                if idx >= max_idx - 1:
                    break
                
        if end_to_end:
            metadata = {'labels': labels if all(
                x is not None for x in labels) else None}
            save_function(os.path.join(destination_path, 'dataset.json'),
                    json.dumps(metadata))
            close_destination()
            sys.exit(0)
            
        return max_idx, iterate_images()
    except Exception as e:
        print(e, labels)
        print(f'Error.......')
        logger.write(f'Error....\n')
        sys.exit(0)

# ----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        from torch import Tensor
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        if isinstance(img, Tensor):
            img = PIL.Image.fromarray(img.numpy())
        else:
            img = PIL.Image.fromarray(img)            
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.Resampling.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) //
                  2, (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
        if img.ndim == 2:
            img = img[:, :, np.newaxis].repeat(3, axis=2)
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2: (img.shape[0] + ch) // 2]
        if img.ndim == 2:
            img = img[:, :, np.newaxis].repeat(3, axis=2)
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2: (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if output_width is None or output_height is None:
            raise click.ClickException(
                'must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if output_width is None or output_height is None:
            raise click.ClickException(
                'must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

# ----------------------------------------------------------------------------

def open_dataset(source, transform_function, destination_path, save_function, close_destination_dir, end_to_end, max_images: Optional[int]):
    if os.path.isdir(source):
        if source.rstrip('/').endswith('_lmdb'):
            return open_lmdb(source, max_images=max_images)
        elif 'multicoil' in os.path.basename(source):    
            return open_mridata(source, transform_function, destination_path, save_function, close_destination_dir, end_to_end, max_images=max_images)
        else:
            return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if os.path.basename(source) == 'cifar-10-python.tar.gz':
            return open_cifar10(source, max_images=max_images)
        elif os.path.basename(source) == 'train-images-idx3-ubyte.gz':
            return open_mnist(source, max_images=max_images)
        elif file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            assert False, 'unknown archive type'
    else:
        raise click.ClickException(
            f'Missing input file or directory: {source}')

# ----------------------------------------------------------------------------


def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w',
                             compression=zipfile.ZIP_STORED)

        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        # if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
        #     raise click.ClickException('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

# ----------------------------------------------------------------------------


@click.command()
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--transform',  help='Input crop/resize mode', metavar='MODE',            type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--resolution', help='Output resolution (e.g., 512x512)', metavar='WxH',  type=parse_tuple)
def main(
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]]
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                    Load LSUN dataset
    --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
    --source train-images-idx3-ubyte.gz Load MNIST dataset
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, class labels are determined from
    top-level directory names.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --resolution option.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --resolution=512x384
    """

    PIL.Image.init()

    if dest == '':
        raise click.ClickException(
            '--dest output filename or directory must not be an empty string')

    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    if resolution is None:
        resolution = (None, None)
    transform_image = make_transform(transform, *resolution)
    end_to_end = False
    if 'multicoil' in source:
        end_to_end = True
        open_dataset(source, transform_image, archive_root_dir, save_bytes, close_dest, end_to_end, max_images=max_images)
    
    num_files, input_iter = open_dataset(source, transform_image, archive_root_dir, save_bytes, close_dest, end_to_end, max_images=max_images)

    dataset_attrs = None

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
        # Apply crop and resize.
        img = transform_image(image['img'])
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1], 'height': img.shape[0], 'channels': channels}
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                raise click.ClickException(
                    f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if dataset_attrs['channels'] not in [1, 3]:
                raise click.ClickException(
                    'Input images must be stored as RGB or grayscale')
            # if width != 2 ** int(np.floor(np.log2(width))):
            #     raise click.ClickException(
            #         'Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [
                f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
            raise click.ClickException(
                f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img, {1: 'L', 3: 'RGB'}[channels])
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname),
                   image_bits.getbuffer())
        labels.append([archive_fname, image['label']]
                      if image['label'] is not None else None)

    metadata = {'labels': labels if all(
        x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'),
               json.dumps(metadata))
    close_dest()

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
