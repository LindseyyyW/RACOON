reduced_label_map = {
    "nobility": [
        "royalty.noble_person",
        "royalty.monarch",
        "royalty.chivalric_order_member",
        "royalty.noble_title"
    ],
    "kingdom": ["royalty.kingdom"],
    "business_operation": ["business.business_operation"],
    "website": ["internet.website"],
    "ethnicity": ["people.ethnicity"],
    "music_artist": [
        "music.writer",
        "music.lyricist",
        "music.producer",
        "music.composer",
        "music.music_video_director",
        "music.artist",
        "music.group_member"
    ],
    "broadcast_artist": ["broadcast.artist"],
    "visual_artist": ["visual_art.visual_artist"],
    "architecture_owner": ["architecture.architectural_structure_owner"],
    "website_owner": ["internet.website_owner"],
    "exhibition_sponsor": ["exhibitions.exhibition_sponsor"],
    "sports_award_winner": ["sports.sports_award_winner"],
    "astronaut": ["spaceflight.astronaut"],
    "music_performance_role": ["music.performance_role"],
    "book_publisher": ["book.periodical_publisher"],
    "human": [
        "people.person",
        "people.family_member",
        "military.military_person",
        "award.hall_of_fame_inductee"
    ],
    "religious_leader": ["religion.religious_leader"],
    "aircraft_owner": ["aviation.aircraft_owner"],
    "government": [
        "government.government_office_or_title",
        "government.governmental_body",
        "government.government_agency"
    ],
    "legislative_session": ["government.legislative_session"],
    "language": ["language.human_language"],
    "organization": [
        "organization.non_profit_organization",
        "award.award_presenting_organization",
        "organization.membership_organization",
        "organization.organization",
        "fictional_universe.fictional_organization"
    ],
    "fraternity_sorority": ["education.fraternity_sorority"],
    "brand": ["business.brand"],
    "music_record_label": ["music.record_label"],
    "currency": ["finance.currency"],
    "industry": ["business.industry"],
    "invention": ["law.invention"],
    "customer": ["business.customer"],
    "sports_event": [
        "tennis.tennis_tournament",
        "soccer.football_world_cup",
        "sports.multi_event_tournament",
        "sports.tournament_event_competition",
        "sports.sports_championship_event",
        "olympics.olympic_games",
        "olympics.olympic_event_competition",
        "sports.sports_league_season",
        "sports.sports_championship",
        "sports.sports_league_draft",
        "soccer.football_league_season"
    ],
    "computer_video_game_creator": ["cvg.cvg_developer", "cvg.cvg_publisher"],
    "computer_video_game": ["cvg.computer_videogame"],
    "computer_video_game_platform": ["cvg.cvg_platform"],
    "sports_position": [
        "ice_hockey.hockey_position",
        "basketball.basketball_position",
        "soccer.football_position",
        "baseball.baseball_position",
        "sports.sports_position"
    ],
    "athlete": [
        "ice_hockey.hockey_player",
        "martial_arts.martial_artist",
        "american_football.football_player",
        "sports.golfer",
        "sports.pro_athlete",
        "cricket.cricket_player",
        "cricket.cricket_bowler",
        "baseball.baseball_player",
        "tennis.tennis_player",
        "chess.chess_player",
        "sports.boxer",
        "sports.cyclist",
        "soccer.football_player",
        "basketball.basketball_player", "tennis.tennis_tournament_champion"
    ],
    "sports_team": [
        "sports.sports_team",
        "sports.professional_sports_team",
        "sports.school_sports_team",
        "american_football.football_team",
        "soccer.football_team",
        "ice_hockey.hockey_team",
        "cricket.cricket_team",
         "baseball.baseball_team",
        "basketball.basketball_team"
    ],
    "sports_league":[
	    "american_football.football_conference", 
        "soccer.football_league", 
        "baseball.baseball_league", 
        "basketball.basketball_conference", 
        "sports.sports_league", 
        "soccer.fifa"
    ],
    "software_license": ["computer.software_license"],
    "software": ["computer.software", "computer.operating_system"],
    "computer": ["computer.computer"],
    "transit_line": ["metropolitan_transit.transit_line"],
    "transit_system": ["metropolitan_transit.transit_system"],
    "transit_stop": ["metropolitan_transit.transit_stop"],
    "product_category": ["business.product_category"],
    "collection_category": ["interests.collection_category"],
    "award_type": [
        "award.award_category",
        "award.award_discipline",
        "sports.sports_award_type"
    ],
    "coach": [
        "american_football.football_coach",
        "basketball.basketball_coach"
    ],
    "football_team_manager": ["soccer.football_team_manager"],
    "actor": [
        "film.actor",
        "tv.tv_actor",
        "theater.theater_actor"
    ],
    "television_production_role": [
        "tv.tv_writer",
        "tv.tv_program_creator",
        "tv.tv_director",
        "tv.tv_producer"
    ],
    "music_composition": ["music.composition"],
    "chemical_element": ["chemistry.chemical_element"],
    "musical_instrument": ["music.instrument"],
    "broadcast": [
        "broadcast.tv_channel",
        "broadcast.broadcast",
        "broadcast.tv_station",
        "broadcast.radio_station"
    ],
    "television_program": [
        "tv.tv_series_season",
        "tv.tv_series_episode",
        "tv.tv_program"
    ],
    "television_network": ["tv.tv_network"],
    "format": [
        "music.media_format",
        "broadcast.radio_format"
    ],
    "genre": [
        "broadcast.genre",
        "media_common.media_genre",
        "tv.tv_genre",
        "film.film_genre",
        "cvg.cvg_genre",
        "music.genre"
    ],
    "tv_personality": ["tv.tv_personality"],
    "celebrity": ["celebrities.celebrity"],
    "film_production_role": [
        "film.writer",
        "film.director",
        "film.producer",
        "film.film_distributor",
        "film.music_contributor"
    ],
    "film": ["film.film"],
    "court": ["law.court"],
    "film_festival_focus": ["film.film_festival_focus"],
    "company": [
        "film.production_company",
        "business.defunct_company",
        "business.consumer_company",
        "automotive.company"
    ],
    "job_title": ["business.job_title"],
    "legal_case": ["law.legal_case"],
    "medicine": [
        "medicine.drug",
        "medicine.medical_treatment",
        "medicine.drug_ingredient"
    ],
    "muscle": ["medicine.muscle"],
    "anatomical_structure": ["medicine.anatomical_structure"],
    "disease": ["medicine.disease"],
    "chemical_compound": ["chemistry.chemical_compound"],
    "astronomical_object": ["astronomy.celestial_object"],
    "astronomical_body": ["astronomy.star_system_body"],
    "constellation": ["astronomy.constellation"],
    "orbital_relationship": ["astronomy.orbital_relationship"],
    "astronomical_discovery": ["astronomy.astronomical_discovery"],
    "sports": ["sports.sport"],
    "military_rank": ["military.rank"],
    "military_unit": ["military.military_unit"],
    "military_armed_force": ["military.armed_force"],
    "military_conflict": ["military.military_conflict"],
    "airport": ["aviation.airport"],
    "airline": ["aviation.airline"],
    "aircraft_model": ["aviation.aircraft_model"],
    "location": [
        "location.province",
        "location.location",
        "location.country",
        "location.us_state",
        "location.australian_state",
        "location.region",
        "location.in_state",
        "location.uk_constituent_country",
        "location.uk_statistical_location",
        "location.hud_county_place",
        "location.citytown",
        "location.us_county",
        "location.capital_of_administrative_division",
        "location.administrative_division",
        "location.australian_local_government_area",
        "location.in_district",
        "location.jp_prefecture",
        "protected_sites.listed_site",
        "military.military_post"
    ],
    "amusement_ride": ["amusement_parks.ride"],
    "amusement_park": ["amusement_parks.park"],
    "election": [
        "government.general_election",
        "government.election",
        "government.election_campaign"
    ],
    "politician": [
        "government.politician",
        "government.u_s_congressperson",
        "government.us_president"
    ],
    "political_party": ["government.political_party"],
    "music_group": ["music.musical_group"],
    "musical_scale": ["music.musical_scale"],
    "album": ["music.album"],
    "play": ["theater.play"],
    "tropical_cyclone": ["meteorology.tropical_cyclone"],
   "tropical_cyclone_season": ["meteorology.tropical_cyclone_season"],
    "educational_institution": [
        "education.school",
        "education.educational_institution",
        "education.university"
    ],
    "educational_degree": ["education.educational_degree"],
    "field_of_study": ["education.field_of_study"],
    "asteroid": ["astronomy.asteroid"],
    "architecture": [
        "architecture.structure",
        "architecture.venue",
        "architecture.building" ],
    "event": [
        "time.event",
        "time.recurring_event",
         "award.competition",
	"award.recurring_competition"
    ],
    "accommodation": ["travel.accommodation"],
    "cricket_stadium": ["cricket.cricket_stadium"],
    "award": [
   "award.award_ceremony",
        "award.award",
        "soccer.football_award"
    ],
    "product": ["business.consumer_product"],
    "organism": [
        "biology.organism_classification",
        "biology.organism"
    ],
    "character": [
        "fictional_universe.fictional_character",
        "comic_books.comic_book_character",
        "film.film_character",
        "tv.tv_character"
    ],
    "athletics_brand": ["education.athletics_brand"],
    "locomotive_class": ["rail.locomotive_class"],
    "author": ["book.author"],
    "book": [
        "book.book",
        "book.written_work"
    ],
    "periodical": [
        "book.magazine",
        "book.periodical",
        "book.newspaper"
    ],
    "periodical_subject": ["book.periodical_subject"],
    "river": ["geography.river"],
    "island": ["geography.island"],
    "body_of_water": ["geography.body_of_water"],
    "mountain": ["geography.mountain"],
    "road": ["transportation.road"],
    "sports_facility": ["sports.sports_facility"],
    "automobile_model": ["automotive.model"],
    "food": ["food.food"],
    "ship": [
        "boats.ship_type",
        "boats.ship"
    ],
    "ship_class": ["boats.ship_class"],
    "religion": ["religion.religion"],
    "animal": ["biology.animal"]
}
reverse_map = {value: key for key, values in reduced_label_map.items() for value in values}
reduced_label_set = reduced_label_map.keys()
reduced_label_set = list(reduced_label_set)