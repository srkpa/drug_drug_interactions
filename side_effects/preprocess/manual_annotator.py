from collections import Counter
from side_effects.expts_results import *
import matplotlib.pyplot as plt
import numpy as np
import csv


def twosides_events_cluster():
    events_clusters = [
        ['cholesterol', 'lipid', 'fat embolism', 'triglycerides', 'hyperlipaemia', 'fatty stools'],
        ['cerebral', 'brain', 'meningeal', 'epilep', 'meningitis', 'dyslexia', 'encephal', 'acalculia',
         'reflex sympathetic dystrophy', 'cephalus', 'cerebellar ataxia', 'mening', 'ataxia', 'dysphasia',
         'astrocytoma', 'neuritis', 'als', 'narcolepsy', 'choreoathetosis', 'cranial', 'arachnoid', 'neuralgia',
         'myelopathy', 'myasthenia', 'apraxia', 'acute porphyria', 'glioma', 'transient ischaemic attack',
         'fibromyalgia', 'blepharospasm', 'amyotrophy', 'petit mal', 'myelitis', 'eeg', 'radiculopathy',
         'glioblastoma multiforme', 'abnormal gait', 'tremor', 'sensory disturbance', 'causalgia', 'agnosia',
         'peripheral nerve injury', 'spinal cord injury', 'brachial plexus injury', 'retrocollis', 'head', 'migraine',
         'ganglion', 'multiple sclerosis', 'near syncope', 'nervous tension'],
        ['cardi', 'heart disease', 'heart attack', 'heart rate', 'heart failure', 'cardio', 'ventricular',
         'palpitation', 'asystole', 'bradycardia', 'tachycardia',
         'arrhythmia', 'afib', 'sinus ', 'tetralogy of fallot', 'beats', 'mitral valve prolapse', 'bundle branch block',
         'ecg', 'post thrombotic syndrome', 'fib ', 'atherosclerosis', 'trifascicular block', 'atrial septal defect',
         'wandering atrial pacemaker', 'mitral valve'],
        ['ear ', 'tympanic membrane', 'hearing', 'otitis', 'deafness', 'labyrinthitis', 'tinnitus', 'cerumen',
         'hyperacusia', 'tympanic',
         'eustachian tube', 'mpd', 'myringitis', 'dysaesthesia'],
        ['skin', 'dermatitis', 'neurodermatitis', 'cutaneous', 'eczema', 'acne', 'scar', 'morphea',
         'pigmentation', 'psoriasis', 'melano', 'erysipelas', 'kerato', 'dermato',
         'depigmentation', 'sunburn', 'xanthoma', 'tinea', 'carbuncle', 'impetigo', 'intertrigo', 'vitiligo',
         'melanoma', 'derma', 'psoriatic', 'erythema', 'cheilosis', 'dandruff', 'dyshidrosis', 'itch', 'rhagades',
         'hyperhidrosis', 'chloasma', 'telangiectases', 'ephelides', 'hirsuitism', 'angiitis', 'angioma',
         'acanthosis nigricans', 'pyogenic granuloma', 'lichen planus', 'hypertrichosis', 'icterus', 'pityriasis',
         'laceration', 'wound', 'xerosis', 'petechia', 'folliculitis', 'ecchymoses', 'bruise', 'pemphigus', 'bleb',
         'milia', 'acantholysis'],
        ['eye', 'retin', 'optic', 'glaucoma', 'blindness', 'corneal', 'vision', 'ectropion', 'entropion', 'cataract',
         'lacrimation', 'lacrimal', 'blepharoptosis', 'macula lutea degeneration', 'chalazion', 'ophthalmo',
         'vitreous detachment', 'hordeolum', 'conjunctivitis', 'myopia', 'pterygium', 'blepharitis', 'anisocoria',
         'asthenopia', 'vitreous haemorrhage', 'iridocyclitis', 'amblyopia', 'strabismus', 'meibomianitis',
         'scleritis', 'hypermetropia', 'scotoma', 'iritis', 'nystagmus', 'floaters', 'light sensitivity', 'intraocular',
         'keratitis', 'refraction disorder', 'dacrocystitis'],
        ['bone', 'rickets', 'osteoporosis', 'coccydynia', 'abdomen', 'abdominoplasty', 'abnormal chest',
         'arthro', 'arthritis', 'joint', 'osteoarthritis', 'gout', 'osteoma', 'ankylosing spondylitis',
         'costochondritis', 'tendinitis', 'spondylitis', 'chondrodystrophy', 'jra', 'osteo', 'scoliosis', 'synovitis',
         'otosclerosis', 'sacroiliitis', 'spondylosis', 'chondrocalcinosis pyrophosphate', 'contracture', 'muscle',
         'cramp', 'bunion', 'pelvic', 'shoulder', 'back ache', 'chest pain', 'musculoskeletal', 'abdominal', 'flank',
         'extremity ', 'neck', 'injury of neck', 'pelvis ', 'spinal fracture', 'rib ', 'femur ', 'tendon ', 'hip ',
         'wrist ', 'humerus ', 'back injury',
         'ankle ', 'intervertebral ', 'disc ', 'intervertebral disc herniation', 'whiplash injury',
         'spinal compression fracture', 'dislocation', 'nonunion', 'cervical', 'trigger finger', 'carpal tunnel'],
        ['walking', 'balance disorder', 'spinning sensation', 'autonomic instability', 'road traffic accident',
         'movement disorder', 'vestibular disorder', 'abasia'],
        ['sleep', 'nocturia', 'nightmare', 'insomnia', 'night sweat', 'abnormal dreams'],
        ['panic', 'tetany', 'hesitancy', 'anxiety', ' stress', 'agitated', 'depression', 'phobia', 'suicide', 'fatigue',
         'hypochondriasis', 'feeling unwell', 'adjustment disorder', 'seasonal affective disorder',
         'bipolar', 'abulia'],
        ['breast', 'mammogram', 'nipple', 'galactorrhea', 'gynaecomastia'],
        ['ovarian', 'vagin', 'uterine', 'ovula', 'infertility', 'menstrual', 'myoma', 'estrogen', 'phimosis',
         'caesarean', 'ejaculation', 'masculinization', 'oligomenorrhea', 'pregnancy', 'testicular', 'volvulus',
         'adenomyosis', 'pregnancies', 'menopause', 'placenta', 'still birth', 'abortion', 'neonatal', 'orchiectomy',
         'endometr', 'hydrocele', 'birth defect', 'eclampsia', 'oophorectomy', 'polymenorrhea',
         'vulv', 'dysmenorrhea', 'hypomenorrhea', 'amenorrhea', 'hot flash', 'pms', 'postmenopausal', 'ovary disease',
         'birth', 'abnormal labour'],
        ['sexually', 'serology run positive', 'hiv', 'hernia', 'papilloma', 'hepatitis', 'herpes', 'orchitis',
         'syphilis', 'gonorrhea', 'penile',
         'genita', 'epididymitis', 'gonad', 'libido', 'salpingitis', 'dyspareunia', 'hematospermia', 'proctitis',
         'herpetic', 'trichomoniasis', 'colpitis', 'mycosis', 'mycosis', 'balanitis', 'cryptorchidism',
         'swollen scrotum', 'peyronies disease', 'testes', 'cervicitis', 'acquired immune deficiency syndrome'],
        ['obesity', 'bulimia', 'weight', 'hyperalimentation', 'food', 'excessive thirst', 'weight', 'anorexia',
         'malnourished', 'eating', 'failure to thrive', 'cachectic'],
        ['cancer', 'polyp', 'neuroblastoma', 'carcinoma', 'sclc', 'tumor', 'laryngeal neoplasia', 'carcinoid',
         'neoplasia',
         'adenoma', 'fibroids', 'phaeochromocytoma', 'neoplasm', 'prolactinoma', 'cyst', 'nodule', 'lipoma', 'sarcoma',
         'neuroma'],
        ['mental', 'hyperaesthesia', 'cognitive disorder', 'neurosis', 'psychos', 'personality disorder', 'excoriation',
         'amentia', 'schizophrenia', 'ekbom', 'delirium', 'cataplexy', 'dysphemia', 'asthenia', 'paraesthesia',
         'hallucination', 'diplopia', 'encopresis', 'enuresis', 'paranoia', 'flashbacks', 'burning sensation',
         'schizoaffective disorder', 'conversion disorder', 'bipolar'],
        ['trisomy 21', 'neurofibromatosis', 'malformation', 'deformity', 'hemophilia', 'porphyria', 'ichthyosis',
         'dwarfism', 'haemophilia'],
        ['kidney', 'glomerulonephritis', 'nephropathy', 'nephrotic', 'nephrosclerosis', 'nephrolithiasis', 'nephritis',
         'rhabdomyolysis', 'nephrocalcinosis'],
        ['diabet', 'insulin', 'hypoglycaemia', 'hyperglycaemia'],
        ['allerg', 'intolerance', 'anaphylactic'],
        ['alcohol', 'drug', 'abuse', 'overdose'],
        ['breath', 'asthma', 'mediastinal disorder', 'alveolitis', 'pulmon', 'respiratory', 'bronch', 'sinusitis',
         'naso', 'rhinorrhea', 'choking',
         'orthopnea', 'epistaxis', 'stridor', 'pneumo', 'sneezing', 'lung', 'dyspnea exertional', 'ild', 'apnea',
         'pleural effusion', 'wheeze', 'hypoxia', 'empyema', 'pleurisy', 'epiglottitis', 'nasal', 'dyspnoea',
         'laryngitis', 'atelectasis', 'chronic obstructive airway disease', 'hypercapnia', 'sarcoidosis', 'emphysema',
         'neumonia', 'hypoventilation', 'adenoid hypertrophy', 'hemothorax', 'anosmia', 'alveolar proteinosis',
         'cough', 'pharyngeal', 'tracheitis', 'tracheostomy', 'pleural pain', 'throat'],
        ['polio', 'parkinson', 'motion', 'hemiparesis', 'quadriplegia', 'amputation', 'hemiplegia',
         'paraplegia', 'movements', 'paraparesis', 'palsies', 'motor retardation', 'facial palsy'],
        ['colon', 'intestin', 'ileus', 'diverticulitis', 'enterocolitis', 'coeliac', 'colitis ischemic', 'duodenitis',
         'diverticulosis', 'enteritis', 'colitis', 'intussusception', 'bowel'],
        ['tooth', 'gingivitis', 'leukoplakia', 'periodontal', 'bruxism', 'periodontitis', 'gingival', 'caries'],
        ['amnesia', 'coma', 'loss of consciousness', 'fainting', 'dizziness', 'drowsiness', 'convulsion',
         'syncope vasovagal', 'confusion', 'near syncope'],
        ['albuminuria', 'proteinuria', 'bladder', 'uropathy', 'dysuria', 'uret', 'renal', 'glucosuria',
         'pyelonephritis', 'papillary necrosis', 'hydronephrosis', 'hydronephrosis', 'menometrorrhagia',
         'priapism', 'micturition', 'varicocele', 'hypospadias', 'urinary', 'urine', 'retroperitoneal fibrosis',
         'polyuria', 'myoglobinuria', 'hemoglobinuria', 'hypercalcinuria', 'pyuria', 'uro'],
        ['hair', 'alopecia'],
        ['furuncle', 'nail', 'onychomycosis', 'onycholysis', 'onych'],
        ['leucopenia', 'leukopenia', 'neutropenia', 'leukaemia', 'platelet', 'leucocytosis', 'myelodysplasia',
         'lymphocyt', 'macrocytosis', 'erythrocyte', 'globuli', 'leukemia', 'Atherosclerosis', 'cell', 'eosinophil',
         'granulocytoses', 'lymphoma', 'erythremia', 'thrombocytopenia', 'myeloma', 'thrombocythemia',
         'monoclonal gammopathy', 'thrombotic microangiopathy', 'granuloma', 'methaemoglobinaemia'],
        ['aphasia', 'vocal', 'phonia', 'dysarthria', 'hoarseness'],
        ['hepatic', 'liver', 'peliosis', 'cirrhosis', 'lfts'],
        ['blood', 'transfusion', 'ischaemia', 'arter', 'vein', 'aneurysm', 'vena', 'vascular',
         'coronary', 'apoplexy', 'leriche syndrome', 'acidose', 'carotid', 'anemia', 'anaemia', 'aort',
         'venous', 'haematoma', 'hematoma', 'phlebitis', 'clotting', 'phlebothrombosis', 'lymphangitis', 'angiopathy',
         'raynauds phenomenon', 'embolism', 'lymphedema', 'adenitis', 'gangrene', 'angina', 'edema', 'seroma',
         'lymphatic disorders'],
        ['esophag', 'odynophagia', 'gastric', 'belching', 'gastritis', 'black stools', 'laparotomy', 'peritonitis',
         'dyspepsia', 'haematochezia', 'malabsorption', 'pyloric stenosis', 'regurgitation', 'emesis', 'nausea',
         'flatulence', 'ascites', 'ulcer', 'deglutition disorder', 'acid reflux', 'stomach', 'abnormal faeces'],
        ['infection', 'infestation', 'tetanus', 'aspergillosis', 'anthrax', 'bacteria', 'tuberculin', 'bactera',
         'giardiasis', 'rhinitis', 'fungal', 'contagio', 'legionella', 'rubella', 'salmonella', 'rabies',
         'diphtheria', 'cryptosporidiosis', 'toxoplasmosis', 'infectious', 'histoplasmosis', 'mumps', 'typhoid',
         'sepsis', 'parotid gland enlargement', 'lyme', 'septic shock', 'pertussis', 'viral', 'streptococcal',
         'cryptococcosis', 'chicken pox', 'adenopathy', 'toxic shock', 'abscess', 'candidiasis', 'flu',
         'virus culture positive'],
        ['rectal', 'diarrhea', 'anal', 'faecal', 'defaecation', 'haemorrhoids', 'constipated', 'gastroenteritis',
         'proctalgia', 'rectum', 'appendicitis', 'colostomy', 'peritoneal', 'appendectomy', 'fecal'],
        ['thyroid', 'goiter'],
        ['prostat'],
        ['cholangitis', 'biliar', 'cholelithiasis'],
        ['hypertension', 'high blood pressure', 'hypotension orthostatic'],
        ['pancrea'],
        ['collagen', 'connective tissue', 'periostitis', 'sjogrens syndrome', 'epicondylitis',
         'soft tissue injuries', 'polymyositis', 'polymyalgia rheumatica', 'fasciitis', 'fibrosis', 'fascitis plantar',
         'rotator cuff syndrome', 'bursitis', 'panniculitis', 'soft tissue', 'plantar'],
        ['glossitis', 'tongue', 'cleft lip', 'mouth', 'salivary gland', 'glossodynia', 'aphthous stomatitis', 'oral',
         'aptyalism', 'palate', 'tonsillar', 'tonsillectomy', 'macroglossia', 'tonsillitis', 'salivation', 'face',
         'facial palsy', 'hyperamylasaemia'],
        ['vitamin', 'folate deficiency'],
        ['body temperature', 'fever', 'heat', 'cold', 'chill'],
        ['fistula'],
        ['spleen', 'splenic', 'splen'],
        ['acidos', 'hyperchloremia', 'hyperphosphatemia', 'aspartate aminotransferase increase', 'excess potassium',
         'adh inappropriate', 'hypermagnesemia', 'dehydration', 'alkalosis', 'hypokalaemia', 'electrolyte disorder',
         'hypernatraemia', 'hypophosphataemia', 'hypomagnesaemia', 'hypochloraemia', 'zinc sulphate'],
        ['herxheimer reaction'], ['hypoaldosteronism', 'hypopituitarism', 'acromegaly', 'hyperprolactinaemia'],
        ['ischias'], ['idiopathic'],
        ['amyloidosis'], ['serum sickness', 'drug hypersensitivity'], ['pain'], ['bleed'],
        ['burns second degree', 'animal bite', 'splinter', 'bite'],
        ['mod', 'lyell', 'easy bruisability', 'corticosteroid therapy', 'abnormal laboratory findings',
         'incontinence', 'bulging', 'flashing lights', 'eruption', 'shift to the left'], ['bilirubinaemia'],
        ['endocrine disorder'],
        ['azotaemia', 'hyperuricaemia']
    ]

    types_of_interest = ['Lipid disorders', 'Neuro-toxicity', 'Cardio toxicity', 'Hearing disorders', 'Skin toxicity',
                         'Eye disorders', 'Musculoskeletal disorders', 'Motor control disorders (minor)',
                         'Sleep disorders',
                         'Emotional and behavioral disorders', 'Breast diseases', 'Gynaecologic disorders',
                         'Sexually transmitted infections', 'Eating disorders', 'Cancers and Tumors',
                         'Mental disorders',
                         'Genetic diseases', 'Kidney diseases', 'Blood Sugar levels fluctuations', 'Allergies',
                         'alcohol and drugs abuse',
                         'Respiratory disorders', 'Motor control disorders (Paralysis)', 'Instestinal disorders',
                         'Teeth diseases', 'loss of consciousness', 'Urinary system disorders', 'Hair toxicity',
                         'Nail infection', 'blood cells diseases', 'Speech disorders',
                         'Liver toxicity', 'blood flow diseases', 'Digestive system disorders',
                         'Parasitic and Bacterial Infections',
                         'Rectal disorders',
                         'Tyroid toxicity',
                         'Prostatic diseases', 'Biliary tract disorders', 'Blood pressure disorders',
                         'Pancreatic diseases',
                         'Soft and Connective tissues disorders',
                         'Mouth and lips diseases',
                         'Vitamin deficiency', 'Fever', 'Fistula', 'Spleen injuries', 'Electrolyte disturbances',
                         'herxheimer reaction', 'Pituitary disorders', 'ischiatic nerve disorders',
                         'Idiopathic diseases',
                         'Bad protein folding disorders', 'Hypersensitivity', 'Pain', 'Bleeding',
                         'General wounds and injuries',
                         'Unclear event report', 'bilirubinaemia', 'endocrine disorders',
                         'nitrogenous deposit in blood']

    assert len(types_of_interest) == len(events_clusters)

    print(len(types_of_interest))
    ddis_groups = dict(zip(types_of_interest, events_clusters))

    return ddis_groups


def event_to_it_type(raw_twosides, ddis_groups):
    distinct_events, types_associated = raw_twosides['event_name'].drop_duplicates(
        keep='first').values.tolist(), []
    types_associated = []
    for event in distinct_events:
        # event_types_associated, find = [], False
        find = False
        for type, values in ddis_groups.items():
            for ev in values:
                if ev in event.lower() + ' ':
                    find = True
                    # event_types_associated.append(type)
                    break
            if find:
                types_associated.append(type)
                print(event, "-", type)
                break
        # if find:
        #     types_associated.append(event_types_associated)
        #     print(event, event_types_associated)
        # else:
        #     print('#{}'.format(event))

    # assert len(distinct_events) == len(types_associated)

    res = dict(zip(distinct_events, types_associated))
    print(len(res))
    return res


def assign_twosidesdrugpairs_event_type(filename='../dataset/files/3003377s-twosides.tsv', fig=False):
    raw_data = pd.read_table(filename, sep='\t')
    groups = twosides_events_cluster()
    res = event_to_it_type(raw_twosides=raw_data, ddis_groups=groups)
    data = defaultdict(list)
    for elem in raw_data[['drug1', 'drug2', 'event_name']].values.tolist():
        data[(elem[0], elem[1])].append(elem[-1])

    final_data = defaultdict(list)
    for pair, events in data.items():
        tmp = []
        for event in events:
            # for x in res[event]:
            #     if x != 'Unclear event report':
            #         tmp.append(x)
            if res[event] != 'Unclear event report':
                tmp.append(res[event])
        if len(tmp) != 0:
            final_data[pair] = list(set(tmp))

    if fig:
        ddis_types_distribution = dict(Counter([event for elem in final_data.values() for event in elem]))
        ind = np.arange(len(ddis_types_distribution.keys()))
        plt.bar(ind, ddis_types_distribution.values(), width=0.85, align='center')
        plt.xticks(ind, ddis_types_distribution.keys(), rotation=90, fontsize=5)
        print(ind)
        plt.title('repartition des classes dans twosides', fontsize=8)
        plt.ylabel('nb_occurences')
        plt.xlabel('ddis types')

        plt.savefig('../figures/twosides-1-v2.png')
        plt.show()

    w = csv.writer(open("../dataset/files/twosides-1-v2.csv", "w"))
    for key, val in final_data.items():
        w.writerow([key[0], key[1], "|".join(val)])
    return final_data


def assign_twosides_drugs_pair_efficacity_type(offside='../dataset/files/3003377s-offsides.tsv',
                                               twosides='../dataset/files/3003377s-twosides.tsv'):
    offsides, twosides = pd.read_table(offside, sep='\t'), pd.read_table(twosides, sep='\t')
    off_data, two_data = defaultdict(list), defaultdict(list)
    for elem in offsides[['drug', 'event']].values.tolist():
        off_data[elem[0].lower()].append(elem[-1].lower())

    for elem in twosides[['drug1', 'drug2', 'event_name']].values.tolist():
        two_data[(elem[0].lower(), elem[1].lower())].append(elem[-1].lower())

    final_data = {}
    for (drug1, drug2) in two_data.keys():
        if drug1 in off_data.keys() and drug2 in off_data.keys():
            cpt1, cpt2 = [True for x in off_data[drug1] if x in two_data[(drug1, drug2)]], [True for x in
                                                                                            off_data[drug2] if
                                                                                            x in two_data[
                                                                                                (drug1, drug2)]]
            print(drug1, drug2, len(cpt1), len(cpt2), len(two_data[(drug1, drug2)]))
            if len(cpt1) + len(cpt2) == len(two_data[(drug1, drug2)]):
                final_data[(drug1, drug2)] = 'additive'
            elif len(cpt1) + len(cpt2) < len(two_data[(drug1, drug2)]):
                final_data[(drug1, drug2)] = 'synergistic'
            else:
                final_data[(drug1, drug2)] = 'antagonistic'
    ind = dict(Counter(final_data.values()))
    print(ind)
    # plot_pie(sizes=list(ind.values()), labels=list(ind.keys()), colors=['lightcoral', 'lightskyblue', 'yellowgreen'],
    #          save_to='../figures/twosides-2.png')

    w = csv.writer(open("../dataset/files/twosides-2.csv", "w"))
    for key, val in final_data.items():
        w.writerow([key[0], key[1], val])
    return final_data

