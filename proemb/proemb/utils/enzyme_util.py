import os
import sys

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

PDB_OBSOLETE_REMAP = {'6fed_C': '6eyd_C',
                      '5jn1_A': '6uzu_A',
                      '5xwq_A': '7fbt_A',
                      '4ror_A': '5ulv_A',
                      '6ihz_A': '7emn_A',
                      '6ihz_B': '7emn_B',
                      '6fed_E': '6eyd_E',
                      '6gt1_C': '6s73_C',
                      '6gt1_D': '6s73_D',
                      '6gt1_B': '6s73_B',
                      '6gt1_C': '6s73_C',
                      '6gt1_A': '6s73_A',
                      '6fed_B': '6eyd_B',
                      '6fed_A': '6eyd_A',
                      '4wto_A': '6kjb_A',
                      '1bvs_D': '7oa5_D',
                      '1bvs_A': '7oa5_A',
                      '1bvs_C': '7oa5_C',
                      '1bvs_E': '7oa5_E',
                      '1bvs_H': '7oa5_H',
                      '5x8o_A': '6lk4_A',
                      '5urr_D': '7mtx_D',
                      '3ohm_B': '7sq2_B',
                      '5hbg_B': '6ahi_B',
                      '5hbg_A': '6ahi_A',
                      '5obl_A': '7obe_A',
                      '5obl_B': '7obe_B',
                      '6fed_D': '6eyd_D',

                      '2i6l_A': '7aqb_A',
                      '2i6l_B': '7aqb_B',
                      '3r5q_A': '7ve3_A'
                      }

MIXED_SPLIT_CHAINS = ['4y84_X', '5l5e_X', '6huu_J', '4qby_J', '4ya9_J', '5mp9_k', '5mpa_k', '3von_E', '3von_b',
                      '3von_p', '3von_i', '6hed_4',
                      '6hec_5', '6he8_4', '6he9_3', '6he7_6', '6he8_k', '6hed_h', '6hea_i', '6hea_h', '6he9_i',
                      '3mg8_I', '4qlq_W', '6huv_I',
                      '5fga_W', '4qby_W', '5mpa_j', '5mp9_j', '5lf1_b', '5lf1_B', '5gjq_j', '1iru_R', '5gjq_k',
                      '5lf0_W', '5m32_I', '5le5_I',
                      '5lf1_I', '5lf3_I', '5gjq_q']

