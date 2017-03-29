#pragma once
#include "Import.hpp"
#include "DataSet.hpp"
#include "File.hpp"
#include "Logging.hpp"

class DataModel
{
public:
	const DataSet& ds;

public:
	vector<tuple<int, int, int>> knowledge;
	vector<vector<int>> description;
	vector<tuple<int, int, int>> ds_train;
	vector<tuple<int, int, int>> ds_valid;
	vector<tuple<int, int, int>> ds_test;
	vector<pair<tuple<int, int, int>, vector<int>>> ds_question;
	map<string, int>	 words;
	vector<string> name;

public:
	int n_entity;
	int n_relation;
	int n_question;
	int n_word;

public:
	DataModel(const DataSet& ds)
		:ds(ds), n_entity(0), n_relation(0), n_question(0)
	{
		logout.record() << "[DataSet] " << ds.name;

		make_fin(ds.url_knowledge) >>knowledge;
		logout.record() << "Knowledge Loaded.";

		make_fin(ds.url_test) >> ds_test;
		logout.record() << "Testing Set Loaded.";

		make_fin(ds.url_valid) >> ds_valid;
		logout.record() << "Validation Set Loaded.";

		for (auto i = knowledge.begin(); i != knowledge.end(); ++i)
		{
			n_entity = max(n_entity, get<0>(*i));
			n_entity = max(n_entity, get<2>(*i));
			n_relation = max(n_relation, get<1>(*i));
		}
		
		make_fin(ds.url_description) >> description;
		logout.record() << "Description Loaded.";

		make_fin(ds.url_word) >> words;
		logout.record() << "Words Set Loaded.";

		n_question = ds_question.size();
		n_word = words.size();

		name.resize(words.size());
		for (auto i = words.begin(); i != words.end(); ++i)
		{
			name[i->second] = i->first;
		}

		logout.record() << "[Data Description] Entity Number = " << n_entity;
		logout.record() << "[Data Description] Relation Number = " << n_relation;
		logout.record() << "[Data Description] Triple Number = " << knowledge.size();
		logout.record() << "[Data Description] Question Number = " << n_question;
		logout.record() << "[Data Description] Word Number = " << n_word;

		logout.record();
	}
};