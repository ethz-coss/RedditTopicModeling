import os
import json
import logging
from steps.zst_io import read_lines_zst
import config
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger('subreddit_extraction')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

temp_file_path_j = '/cluster/work/coss/anmusso/victoria/temp.json'
temp_file_path_p = '/cluster/work/coss/anmusso/victoria/temp.parquet'

non_eng_subr = ['15m', 'aachen', 'aalborg', 'aarhus', 'afrikaans', 'aguascalientes', 'aixmarseille', 'albania', 'alltfaller', 'antwerpen/', 'anunturi', 'arautos', 'argaming', 'argenbeauty', 'argenbitcoin', 'argencirclejerk', 'argentina', 'argentinacirclejerk', 'argentinacocina', 'argentos', 'arggw', 'arteperuano', 'ateosmexicanos', 'atomausstieg', 'austria', 'aveiro', 'bahiablanca', 'bajacalifornia', 'bajacaliforniasur', 'benfica', 'bergen', 'bern', 'bitcoinde', 'bitcoindk', 'bitcoinfrance', 'bitcoinis', 'bitcoinmex', 'bitcoinnl', 'bitcoinno', 'bitcoinswe', 'bitcoinvzla', 'bocajuniors', 'bolivia', 'bonn', 'bopvn', 'boras', 'bordeaux', 'bossanova', 'bostad', 'brasil', 'brasilbitcoin', 'brasildob', 'brasilia', 'braziu', 'bremen', 'bretagne', 'btces', 'buenosaires', 'buenzli', 'bundesliga', 'cameroon', 'cariverplate', 'cflcomics', 'chairos', 'chile', 'chilecringe', 'chinesereddit', 'chistes', 'ciencia', 'cinefilos', 'circuloidiota', 'cluj', 'coimbra', 'cologne', 'colombia', 'cordoba', 'coyoacan', 'cpop', 'croatia', 'cryptocroatia', 'cryptomexico', 'danish', 'danishents', 'danskrap', 'darkmarketsbrasil', 'darknetmarketsno', 'darmstadt', 'datenschutz', 'de', 'de_comedy', 'de_iama', 'de_it', 'de_podcasts', 'de_punkq', 'demandezreddit', 'denmark', 'dentacoin', 'deohneregeln', 'desporto', 'deutsche_queers', 'deutschland', 'diepartei', 'distritofederal', 'djangoes', 'dkmusik', 'dkpol', 'dkwork', 'dogecoingermany', 'dokumentationen', 'dominican', 'dresden', 'easy_german', 'ecouteca', 'ecuador', 'edomex', 'einfach_posten', 'eintracht', 'elsalvador', 'eltern', 'empleos_ar', 'ensenada', 'entraideinformatique', 'erba', 'es', 'espanol', 'estadodemexico', 'estadosunidos', 'ethtradernl', 'etsmtl', 'explicacomoanino', 'fceyn', 'fcsp', 'feministas', 'fernsehen', 'finanzen', 'firenze', 'fitnessargentina', 'forwardsdemama', 'fr', 'fragreddit', 'france', 'francesinhas', 'frankfurt', 'freiburg', 'fremdscham', 'frenchimmersionportuguese', 'frenchrap', 'fulbo', 'futbol', 'futbolmx', 'gamingesp', 'gavle', 'genova', 'gent', 'germanimmersion', 'germanmovies', 'germanpractice', 'germanpuns', 'germanrap', 'germantrees', 'germanvideos', 'germanyusa', 'germusic', 'goe', 'gothenburg', 'grandsud', 'graz', 'groningen', 'guadalajara', 'guanajuato', 'guatemala', 'halmstad', 'hamburg', 'hannover', 'heidelberg', 'helsinki', 'hindi', 'hoerbuecher', 'hola', 'honduras', 'hungary', 'iasi', 'iceland', 'ilandsproblem', 'indonesia', 'interlingua', 'investimentos', 'italianlearning', 'italians', 'italy', 'italy_ss', 'italyinformatica', 'ja', 'jagverse', 'japaneseonly', 'japanesereddit', 'jememarre', 'jeuxvideo', 'jobb', 'joppa', 'karlsruhe', 'karlstad', 'knugencoin', 'kochen', 'koeln', 'kolding', 'kopiernudeln', 'krakow', 'kreiswichs', 'kripto', 'lalaguna', 'laliga', 'laroja', 'latinoamerica', 'latvia', 'learngerman', 'lecafe', 'leipzig', 'lgbtq_de_argentina', 'ligamx', 'ligue1', 'lille', 'linkoping', 'linz', 'lisboa', 'litigi', 'livres', 'lodz', 'lordsofjammore', 'lund', 'lyon', 'madrid', 'malmo', 'manele', 'marknad', 'mejorde', 'memexico', 'mercadoreddit', 'metaswirclejerk', 'mexicali', 'mexico', 'mexicocirclejerk', 'mexicoiama', 'mexijobs', 'milano', 'monterrey', 'montpellier', 'montreal', 'montrealfood', 'montrealgay', 'montrealjobs', 'morocco', 'musica', 'musicaenespanol', 'muzyka', 'nahuatl', 'nantes', 'napoli', 'nederland', 'nederlands', 'nerdsmexico', 'nicaragua', 'nichtderpostillon', 'niedersachsen', 'nihon', 'nihongo', 'nijmegen', 'nl', 'norge', 'notargentinar', 'noticias', 'ntnu', 'oaxaca', 'obitcoin', 'occuponsmontreal', 'occuponsmontrealag', 'occuponsmontrealjobs', 'odense', 'oradea', 'orebro', 'oslo', 'panama', 'paraguay', 'peru', 'photosmontreal', 'pietsmiet', 'pikabu', 'pikabushort', 'piratenpartei', 'pitesti', 'playa', 'podemos', 'polityka', 'polska', 'polymtl', 'porto', 'portugal', 'portugalnews', 'poznan', 'practicar', 'programacion', 'programmation', 'programmazione', 'provence', 'puertorico', 'quebec', 'queretaro', 'quintanaroo', 'r4rmontreal', 'ragefrance', 'rance', 'ratschlagtiere', 'rayados', 'rbtvcirclejerk', 'rebootfrance', 'recht', 'reddist', 'redditores', 'riograndedosul', 'rocirclejerk', 'rocketbeans', 'role', 'rolistentothis', 'romania', 'romania_ss', 'rosario', 'rosarito', 'ru', 'ruhrgebiet', 'russiafunny', 'russianimmersion', 'santiago', 'saopaulo', 'schalke04', 'schland', 'schleswigholstein', 'schweiz', 'seilbahn', 'self_de', 'serbia', 'simpsonsespanollatino', 'sinaloa', 'skellefte', 'soluzioni', 'sonora', 'spain', 'spanishimmersion', 'spel', 'spvm', 'squareposting', 'starcraftdeutschland', 'startrekgermany', 'stavanger', 'stockholm', 'suisse', 'suomi', 'svenska', 'swadviceanimals', 'sweden', 'swedents', 'swiama', 'swirclejerk', 'switzerland', 'tampere', 'teutonik', 'the_frauke', 'thenetherlands', 'ticos', 'tigres', 'tijuana', 'timisoara', 'torino', 'toulouse', 'trondheim', 'truefrance', 'tuebingen', 'tugajerk', 'turkey', 'turkeyjerky', 'tvadicto', 'udem', 'udes', 'ukraina', 'ulaval', 'umea', 'uppsala', 'uqam', 'uruguay', 'uruguaysoccer', 'valencia', 'vasteras', 'vencirclejerk', 'venezia', 'veracruz', 'vfbstuttgart', 'videojuego', 'vlaanderen', 'vosfinances', 'vzla', 'warsaw', 'weibsvolk', 'wien', 'wiesbaden', 'wirtschaft', 'wissenschaft', 'yo_elvr', 'yucatan', 'zacatecas', 'zocken']


def _filter_comment(line_json):
    if ((config.FILTER == True) and (len(line_json['body']) > 20)) or (config.FILTER == False):
        desired_attributes = {'score', 'subreddit', 'author', 'created_utc', 'id',
                              'parent_id'}  # Specify the attributes you want to keep
        filtered = {key: value for key, value in line_json.items() if key in desired_attributes}
        filtered['title'] = line_json['body']  # rename to create consistency with submissions
        return filtered
    else:
        return None


def _filter_submission(line_json):
    b = False

    if ((config.FILTER == False) and (len(line_json['title'])) > 20) and (line_json['media'] is None):
        b = True
    if (config.FILTER == True) and (len(line_json['title']) > 20) and (line_json['media'] is None) and \
            (line_json['score'] > 20) and (line_json['num_comments'] > 10):
        b = True

    if b:
        desired_attributes = {'subreddit', 'score', 'author', 'created_utc', 'title', 'id', 'num_comments',
                              'upvote_ratio'}  # Specify the attributes you want to keep
        return {key: value for key, value in line_json.items() if key in desired_attributes}
    else:
        return None


def _create_comments_table(table_name, sql_db):
    sql_db.sql(f"DROP TABLE IF EXISTS {table_name}")
    sql_db.sql(
        f"CREATE TABLE {table_name} (NUM INT PRIMARY KEY , score INT, subreddit VARCHAR, author VARCHAR, created_utc INTEGER, "
        "title VARCHAR , id VARCHAR, parent_id VARCHAR )")


def _create_submissions_table(table_name, sql_db):
    sql_db.sql(f"DROP TABLE IF EXISTS {table_name}")
    sql_db.sql(
        f"CREATE TABLE {table_name} (NUM INT PRIMARY KEY, subreddit VARCHAR, score INT, author VARCHAR, "
        "created_utc INTEGER, title VARCHAR, id VARCHAR, num_comments INTEGER, upvote_ratio DOUBLE )")


def _add_to_db_parquet(good_lines, table_name,
                       sql_db):  #maybe this isnt working because i need to get all the dicts into the same oder first
    tables = []
    with open(temp_file_path_p, 'wb') as f:
        for d in good_lines:
            table = pa.Table.from_pydict({key: [d[key]] for key in d})
            tables.append(table)
        # Write the final_table to a Parquet file
        pq.write_table(table=pa.concat_tables(tables), where=f)
        #pa.parquet.write_table(pa.concat_tables(tables), f)

    sql_db.execute(f"COPY {table_name} FROM '{temp_file_path_p}' (FORMAT PARQUET);")
    print("added to collection", len(good_lines))


def _add_to_db_json(good_lines, table_name, sql_db):
    with open(temp_file_path_j, 'w') as f:
        json.dump(good_lines, f)
    sql_db.execute(
        f"COPY {table_name} FROM '{temp_file_path_j}' (FORMAT JSON, AUTO_DETECT true) ;")
    #print("added to collection", len(good_lines))


#give option to create new table or not
def extract_comments(input_file_name: str, sql_db, table_name: str):
    reader_chunk_size: int = 2 ** 27
    reader_window_size: int = 2 ** 31
    good_lines = []

    if ((table_name,) in sql_db.execute("SHOW TABLES;").fetchall()):
        max_id = sql_db.sql(f"SELECT MAX(num) FROM {table_name}").fetchone()
        if max_id[0] is not None:
            good_lines_count = max_id[0] + 1
        else:
            good_lines_count = 0
    else:
        _create_comments_table(table_name, sql_db)
        good_lines_count = 0

    #initialize logger variables
    bad_lines, file_lines, file_bytes_processed = 0, 0, 0
    file_size = os.stat(input_file_name).st_size

    # Loop through every line in the file
    for line, file_bytes_processed in read_lines_zst(file_name=input_file_name, reader_window_size=reader_window_size,
                                                     reader_chunk_size=reader_chunk_size):
        try:
            line_json = json.loads(line)
            filtered_line = _filter_comment(line_json)  #do basic preselection of lines
            if filtered_line is not None:
                filtered_line['NUM'] = good_lines_count
                good_lines_count += 1
                good_lines.append(filtered_line)  #add to list of valid lines
        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1
        file_lines += 1

        # Log progress
        if file_lines % 400_000 == 0:
            logger.info(
                f": {good_lines_count:,} {file_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%")

        # Write the lines to the db file
        if len(good_lines) > 50_000:  #for now only opening json file once is fastest..
            _add_to_db_json(good_lines, table_name, sql_db)
            good_lines = []

    #add last bit of lines
    _add_to_db_json(good_lines, table_name, sql_db)

    print('total added lines: ', sql_db.sql(f"SELECT COUNT(*) FROM {table_name}"))
    logger.info(f"Complete : {file_lines:,} : {bad_lines:,} : {good_lines_count:,}")


def extract_submissions(input_file_name: str, sql_db, table_name: str):
    reader_chunk_size: int = 2 ** 27
    reader_window_size: int = 2 ** 31
    good_lines = []

    if ((table_name,) in sql_db.execute("SHOW TABLES;").fetchall()):
        max_id = sql_db.sql(f"SELECT MAX(num) FROM {table_name}").fetchone()
        if max_id[0] is not None:
            good_lines_count = max_id[0] + 1
        else:
            good_lines_count = 0
    else:
        _create_submissions_table(table_name, sql_db)
        good_lines_count = 0
    #initialize tracking variables
    bad_lines, file_lines, file_bytes_processed = 0, 0, 0
    file_size = os.stat(input_file_name).st_size

    # Loop through every line in the file
    for line, file_bytes_processed in read_lines_zst(file_name=input_file_name, reader_window_size=reader_window_size,
                                                     reader_chunk_size=reader_chunk_size):
        try:
            line_json = json.loads(line)
            filtered_line = _filter_submission(line_json)  #do basic preselection of lines
            if filtered_line is not None:
                filtered_line['NUM'] = good_lines_count
                good_lines_count += 1
                good_lines.append(filtered_line)  #add to list of valid lines

        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1
        file_lines += 1

        # Log progress
        if file_lines % 200_000 == 0:
            logger.info(
                f": {good_lines_count:,} {file_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%")

        # Write the lines to the db file
        if len(good_lines) > 10_000:  #for now only opening json file once is fastest..
            _add_to_db_json(good_lines, table_name, sql_db)
            good_lines = []

    #add last bit of lines
    _add_to_db_json(good_lines, table_name, sql_db)

    print('total added lines: ', sql_db.sql(f"SELECT COUNT(*) FROM {table_name}"))
    logger.info(f"Complete : {file_lines:,} : {bad_lines:,} : {good_lines_count:,}")
