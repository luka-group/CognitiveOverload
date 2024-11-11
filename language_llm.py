# Created by xunannancy at 2024/11/04
from collections import OrderedDict

language_llm_dict = {
    'bg': {
        'full_name': 'Bulgarian',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'ca': {
        'full_name': 'Catalan',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'cs': {
        'full_name': 'Czech',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'da': {
        'full_name': 'Danish',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'de': {
        'full_name': 'German',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'es': {
        'full_name': 'Spanish',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'fr': {
        'full_name': 'French',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'hr': {
        'full_name': 'Croatian',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'hu': {
        'full_name': 'Hungarian',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'it': {
        'full_name': 'Italian',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'nl': {
        'full_name': 'Dutch',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'pl': {
        'full_name': 'Polish',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'pt': {
        'full_name': 'Portuguese',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'ro': {
        'full_name': 'Romanian',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'ru': {
        'full_name': 'Russian',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'sl': {
        'full_name': 'Slovenian',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'sr': {
        'full_name': 'Serbian',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'sv': {
        'full_name': 'Swedish',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'uk': {
        'full_name': 'Ukrainian',
        'llms': ['vicuna', 'llama2', 'chatgpt', 'bard'],
    },
    'zh-CN': {
        'full_name': 'Chinese (Simplified)',
        'llms': ['llama2', 'chatgpt', 'bard'],
    },
    'zh-TW': {
        'full_name': 'Chinese (Traditional)',
        'llms': ['llama2', 'chatgpt', 'bard'],
    },
    'ja': {
        'full_name': 'Japanese',
        'llms': ['llama2', 'chatgpt', 'bard'],
    },
    'vi': {
        'full_name': 'Vietnamese',
        'llms': ['llama2', 'chatgpt', 'bard'],
    },
    'ko': {
        'full_name': 'Korean',
        'llms': ['llama2', 'chatgpt', 'bard'],
    },
    'id': {
        'full_name': 'Indonesian',
        'llms': ['llama2', 'chatgpt', 'bard'],
    },
    'fi': {
        'full_name': 'Finnish',
        'llms': ['llama2', 'chatgpt', 'bard'],
    },
    'no': {
        'full_name': 'Norwegian',
        'llms': ['llama2', 'chatgpt', 'bard'],
    },
    'af': {
        'full_name': 'Afrikaans',
        'llms': ['chatgpt'],
    },
    'el': {
        'full_name': 'Greek',
        'llms': ['chatgpt', 'bard'],
    },
    'lv': {
        'full_name': 'Latvian',
        'llms': ['chatgpt', 'bard'],
    },
    'ar': {
        'full_name': 'Arabic',
        'llms': ['chatgpt', 'bard'],
    },
    'tr': {
        'full_name': 'Turkish',
        'llms': ['chatgpt', 'bard'],
    },
    'sw': {
        'full_name': 'Swahili',
        'llms': ['chatgpt', 'bard'],
    },
    'cy': {
        'full_name': 'Welsh',
        'llms': ['chatgpt'],
    },
    'is': {
        'full_name': 'Icelandic',
        'llms': ['chatgpt'],
    },
    'bn': {
        'full_name': 'Bengali',
        'llms': ['chatgpt', 'bard'],
    },
    'ur': {
        'full_name': 'Urdu',
        'llms': ['chatgpt', 'bard'],
    },
    'ne': {
        'full_name': 'Nepali',
        'llms': ['chatgpt'],
    },
    'th': {
        'full_name': 'Thai',
        'llms': ['chatgpt', 'bard'],
    },
    'pa': {
        'full_name': 'Punjabi',
        'llms': ['chatgpt'],
    },
    'mr': {
        'full_name': 'Marathi',
        'llms': ['chatgpt', 'bard'],
    },
    'te': {
        'full_name': 'Telugu',
        'llms': ['chatgpt', 'bard'],
    },
    'et': {
        'full_name': 'Estonian',
        'llms': ['chatgpt', 'bard'],
    },
    'fa': {
        'full_name': 'Persian',
        'llms': ['chatgpt', 'bard'],
    },
    'gu': {
        'full_name': 'Gujarati',
        'llms': ['chatgpt', 'bard'],
    },
    'he': {
        'full_name': 'Hebrew',
        'llms': ['chatgpt', 'bard'],
    },
    'hi': {
        'full_name': 'Hindi',
        'llms': ['chatgpt', 'bard'],
    },
    'kn': {
        'full_name': 'Kannada',
        'llms': ['chatgpt', 'bard'],
    },
    'lt': {
        'full_name': 'Lithuanian',
        'llms': ['chatgpt', 'bard'],
    },
    'ml': {
        'full_name': 'Malayalam',
        'llms': ['chatgpt', 'bard'],
    },
    'sk': {
        'full_name': 'Slovak',
        'llms': ['chatgpt', 'bard'],
    },
    'ta': {
        'full_name': 'Tamil',
        'llms': ['chatgpt', 'bard'],
    }

}

model_lan_dict = {
    'vicuna-7b': ['en'],
    'vicuna-13b':  ['en'],
    'llama2-7b-chat': ['en'],
    'llama2-13b-chat': ['en'],
    'chatgpt': ['en'],
}

for lan, info in language_llm_dict.items():
    if 'vicuna' in info['llms']:
        model_lan_dict['vicuna-7b'].append(lan)
        model_lan_dict['vicuna-13b'].append(lan)
    if 'llama2' in info['llms']:
        model_lan_dict['llama2-7b-chat'].append(lan)
        model_lan_dict['llama2-13b-chat'].append(lan)
    if 'chatgpt' in info['llms']:
        model_lan_dict['chatgpt'].append(lan)
model_lan_dict['mpt-7b-instruct'] = model_lan_dict['vicuna-7b']
model_lan_dict['mpt-7b-chat'] = model_lan_dict['vicuna-7b']
model_lan_dict['guanaco-7b'] = model_lan_dict['vicuna-7b']
model_lan_dict['guanaco-13b'] = model_lan_dict['vicuna-7b']
model_lan_dict['WizardLM-7B-V1.0'] = model_lan_dict['vicuna-7b']
model_lan_dict['WizardLM-13B-V1.2'] = model_lan_dict['vicuna-7b']
model_lan_dict['gpt-3.5-turbo-0301'] = model_lan_dict['chatgpt']


language_llm_name_code_dict = dict()
for code, info in language_llm_dict.items():
    language_llm_name_code_dict[info['full_name']] = code

flores_lan_code_dict = {
    "Acehnese (Arabic script)": "ace_Arab",
    "Acehnese (Latin script)": "ace_Latn",
    "Mesopotamian Arabic": "acm_Arab",
    "Ta’izzi-Adeni Arabic": "acq_Arab",
    "Tunisian Arabic": "aeb_Arab",
    "Afrikaans": "afr_Latn",
    "South Levantine Arabic": "ajp_Arab",
    "Akan": "aka_Latn",
    "Amharic": "amh_Ethi",
    "North Levantine Arabic": "apc_Arab",
    "Modern Standard Arabic": "arb_Arab",
    "Modern Standard Arabic (Romanized)": "arb_Latn",
    "Najdi Arabic": "ars_Arab",
    "Moroccan Arabic": "ary_Arab",
    "Egyptian Arabic": "arz_Arab",
    "Assamese": "asm_Beng",
    "Asturian": "ast_Latn",
    "Awadhi": "awa_Deva",
    "Central Aymara": "ayr_Latn",
    "South Azerbaijani": "azb_Arab",
    "North Azerbaijani": "azj_Latn",
    "Bashkir": "bak_Cyrl",
    "Bambara": "bam_Latn",
    "Balinese": "ban_Latn",
    "Belarusian": "bel_Cyrl",
    "Bemba": "bem_Latn",
    "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva",
    "Banjar (Arabic script)": "bjn_Arab",
    "Banjar (Latin script)": "bjn_Latn",
    "Standard Tibetan": "bod_Tibt",
    "Bosnian": "bos_Latn",
    "Buginese": "bug_Latn",
    "Bulgarian": "bul_Cyrl",
    "Catalan": "cat_Latn",
    "Cebuano": "ceb_Latn",
    "Czech": "ces_Latn",
    "Chokwe": "cjk_Latn",
    "Central Kurdish": "ckb_Arab",
    "Crimean Tatar": "crh_Latn",
    "Welsh": "cym_Latn",
    "Danish": "dan_Latn",
    "German": "deu_Latn",
    "Southwestern Dinka": "dik_Latn",
    "Dyula": "dyu_Latn",
    "Dzongkha": "dzo_Tibt",
    "Greek": "ell_Grek",
    "English": "eng_Latn",
    "Esperanto": "epo_Latn",
    "Estonian": "est_Latn",
    "Basque": "eus_Latn",
    "Ewe": "ewe_Latn",
    "Faroese": "fao_Latn",
    "Fijian": "fij_Latn",
    "Finnish": "fin_Latn",
    "Fon": "fon_Latn",
    "French": "fra_Latn",
    "Friulian": "fur_Latn",
    "Nigerian Fulfulde": "fuv_Latn",
    "Scottish Gaelic": "gla_Latn",
    "Irish": "gle_Latn",
    "Galician": "glg_Latn",
    "Guarani": "grn_Latn",
    "Gujarati": "guj_Gujr",
    "Haitian Creole": "hat_Latn",
    "Hausa": "hau_Latn",
    "Hebrew": "heb_Hebr",
    "Hindi": "hin_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Croatian": "hrv_Latn",
    "Hungarian": "hun_Latn",
    "Armenian": "hye_Armn",
    "Igbo": "ibo_Latn",
    "Ilocano": "ilo_Latn",
    "Indonesian": "ind_Latn",
    "Icelandic": "isl_Latn",
    "Italian": "ita_Latn",
    "Javanese": "jav_Latn",
    "Japanese": "jpn_Jpan",
    "Kabyle": "kab_Latn",
    "Jingpho": "kac_Latn",
    "Kamba": "kam_Latn",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic script)": "kas_Arab",
    "Kashmiri (Devanagari script)": "kas_Deva",
    "Georgian": "kat_Geor",
    "Central Kanuri (Arabic script)": "knc_Arab",
    "Central Kanuri (Latin script)": "knc_Latn",
    "Kazakh": "kaz_Cyrl",
    "Kabiyè": "kbp_Latn",
    "Kabuverdianu": "kea_Latn",
    "Khmer": "khm_Khmr",
    "Kikuyu": "kik_Latn",
    "Kinyarwanda": "kin_Latn",
    "Kyrgyz": "kir_Cyrl",
    "Kimbundu": "kmb_Latn",
    "Northern Kurdish": "kmr_Latn",
    "Kikongo": "kon_Latn",
    "Korean": "kor_Hang",
    "Lao": "lao_Laoo",
    "Ligurian": "lij_Latn",
    "Limburgish": "lim_Latn",
    "Lingala": "lin_Latn",
    "Lithuanian": "lit_Latn",
    "Lombard": "lmo_Latn",
    "Latgalian": 'ltg_Latn',
    'Luxembourgish': 'ltz_Latn',
    'Luba-Kasai': 'lua_Latn',
    'Ganda': 'lug_Latn',
    'Luo': 'luo_Latn',
    'Mizo': 'lus_Latn',
    'Standard Latvian': 'lvs_Latn',
    'Magahi': 'mag_Deva',
    'Maithili': 'mai_Deva',
    'Malayalam': 'mal_Mlym',
    'Marathi': 'mar_Deva',
    'Minangkabau (Arabic script)': 'min_Arab',
    'Minangkabau (Latin script)': 'min_Latn',
    'Macedonian': 'mkd_Cyrl',
    'Plateau Malagasy': 'plt_Latn',
    'Maltese': 'mlt_Latn',
    'Meitei (Bengali script)': 'mni_Beng',
    'Halh Mongolian': 'khk_Cyrl',
    'Mossi': 'mos_Latn',
    'Maori': 'mri_Latn',
    'Burmese': 'mya_Mymr',
    'Dutch': 'nld_Latn',
    'Norwegian Nynorsk': 'nno_Latn',
    'Norwegian Bokmål': 'nob_Latn',
    'Nepali': 'npi_Deva',
    'Northern Sotho': 'nso_Latn',
    'Nuer': 'nus_Latn',
    'Nyanja': 'nya_Latn',
    'Occitan': 'oci_Latn',
    'West Central Oromo': 'gaz_Latn',
    'Odia': 'ory_Orya',
    'Pangasinan': 'pag_Latn',
    'Eastern Panjabi': 'pan_Guru',
    'Papiamento': 'pap_Latn',
    'Western Persian': 'pes_Arab',
    'Polish': 'pol_Latn',
    'Portuguese': 'por_Latn',
    'Dari': 'prs_Arab',
    'Southern Pashto': 'pbt_Arab',
    'Ayacucho Quechua': 'quy_Latn',
    'Romanian': 'ron_Latn',
    'Rundi': 'run_Latn',
    'Russian': 'rus_Cyrl',
    'Sango': 'sag_Latn',
    'Sanskrit': 'san_Deva',
    'Santali': 'sat_Olck',
    'Sicilian': 'scn_Latn',
    'Shan': 'shn_Mymr',
    'Sinhala': 'sin_Sinh',
    'Slovak': 'slk_Latn',
    'Slovenian': 'slv_Latn',
    'Samoan': 'smo_Latn',
    'Shona': 'sna_Latn',
    'Sindhi': 'snd_Arab',
    'Somali': 'som_Latn',
    'Southern Sotho': 'sot_Latn',
    'Spanish': 'spa_Latn',
    'Tosk Albanian': 'als_Latn',
    'Sardinian': 'srd_Latn',
    'Serbian': 'srp_Cyrl',
    'Swati': 'ssw_Latn',
    'Sundanese': 'sun_Latn',
    'Swedish': 'swe_Latn',
    'Swahili': 'swh_Latn',
    'Silesian': 'szl_Latn',
    'Tamil': 'tam_Taml',
    'Tatar': 'tat_Cyrl',
    'Telugu': 'tel_Telu',
    'Tajik': 'tgk_Cyrl',
    'Tagalog': 'tgl_Latn',
    'Thai': 'tha_Thai',
    'Tigrinya': 'tir_Ethi',
    'Tamasheq (Latin script)': 'taq_Latn',
    'Tamasheq (Tifinagh script)': 'taq_Tfng',
    'Tok Pisin': 'tpi_Latn',
    'Tswana': 'tsn_Latn',
    'Tsonga': 'tso_Latn',
    'Turkmen': 'tuk_Latn',
    'Tumbuka': 'tum_Latn',
    'Turkish': 'tur_Latn',
    'Twi': 'twi_Latn',
    'Central Atlas Tamazight': 'tzm_Tfng',
    'Uyghur': 'uig_Arab',
    'Ukrainian': 'ukr_Cyrl',
    'Umbundu': 'umb_Latn',
    'Urdu': 'urd_Arab',
    'Northern Uzbek': 'uzn_Latn',
    'Venetian': 'vec_Latn',
    'Vietnamese': 'vie_Latn',
    'Waray': 'war_Latn',
    'Wolof': 'wol_Latn',
    'Xhosa': 'xho_Latn',
    'Eastern Yiddish': 'ydd_Hebr',
    'Yoruba': 'yor_Latn',
    'Yue Chinese': 'yue_Hant',
    'Chinese (Simplified)': 'zho_Hans',
    'Chinese (Traditional)': 'zho_Hant',
    'Standard Malay': 'zsm_Latn',
    'Zulu': 'zul_Latn',
}

flores_code_llm_code_dict = OrderedDict({
    '__label__eng_Latn': 'en',
})
for name, llm_code in language_llm_name_code_dict.items():
    if name == 'Norwegian':
        flores_code_llm_code_dict[f'__label__{flores_lan_code_dict["Norwegian Nynorsk"]}'] = llm_code
        flores_code_llm_code_dict[f'__label__{flores_lan_code_dict["Norwegian Bokmål"]}'] = llm_code
    elif name == 'Latvian':
        flores_code_llm_code_dict[f'__label__{flores_lan_code_dict["Standard Latvian"]}'] = llm_code
    elif name == 'Arabic':
        for candidate in ['Acehnese (Arabic script)', 'Mesopotamian Arabic', 'Ta’izzi-Adeni Arabic', 'Tunisian Arabic',
                          'South Levantine Arabic', 'North Levantine Arabic', 'Modern Standard Arabic', 'Modern Standard Arabic (Romanized)',
                          'Najdi Arabic', 'Moroccan Arabic', 'Egyptian Arabic', 'Banjar (Arabic script)', 'Kashmiri (Arabic script)',
                          'Central Kanuri (Arabic script)', 'Minangkabau (Arabic script)']:
            flores_code_llm_code_dict[f'__label__{flores_lan_code_dict[candidate]}'] = llm_code
    elif name == 'Punjabi':
        flores_code_llm_code_dict[f'__label__{flores_lan_code_dict["Eastern Panjabi"]}'] = llm_code
    elif name == 'Persian':
        flores_code_llm_code_dict[f'__label__{flores_lan_code_dict["Western Persian"]}'] = llm_code
    else:
        flores_code_llm_code_dict[f'__label__{flores_lan_code_dict[name]}'] = llm_code

llm_coded_flores_code_dict = dict()
for flores_code, llm_code in flores_code_llm_code_dict.items():
    if llm_code in llm_coded_flores_code_dict:
        continue
    llm_coded_flores_code_dict[llm_code] = flores_code[len('__label__'):]