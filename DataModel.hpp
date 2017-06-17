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
	vector<tuple<int, int, int>>						knowledge;
	vector<vector<int>>									description;
	vector<pair<tuple<int, int, int>, vector<int>>>		ds_question_train;
	vector<pair<tuple<int, int, int>, vector<int>>>		ds_question_dev;
	vector<pair<tuple<int, int, int>, vector<int>>>		ds_question_test;
	map<string, int>									words;
	map<int, int>										entity;
	map<int, int>										relation;
	vector<string>										name;

public:
	map<int, map<int, vector<int> > >   rel_heads;
	map<int, map<int, vector<int> > >   rel_tails;
	vector<double>						relation_tph;
	vector<double>						relation_hpt;
	set<tuple<int, int, int>>			sampling_checker;

public:
	int n_entity;
	int n_relation;
	int n_word;

public:
	DataModel(const DataSet& ds)
		:ds(ds)
	{
		logout.record() << "[DataSet] " << ds.name;
		auto fin = make_fin(ds.url);

		fin >> ds_question_train;
		fin >> ds_question_dev;
		fin >> ds_question_test;
		logout.record() << "[Dataset] Questions Loaded.";

		fin >> knowledge;
		
		for (auto i = knowledge.begin(); i != knowledge.end(); ++i)
		{
			sampling_checker.insert(*i);
		}

		fin >> rel_heads;
		fin >> rel_tails;
		fin >> relation_tph;
		fin >> relation_hpt;
		logout.record() << "[Dataset] Knowledge Loaded.";

		fin >> description;
		logout.record() << "[Dataset] Description Loaded.";

		fin >> name;
		logout.record() << "[Dataset] Name Loaded.";

		fin >> n_word;
		fin >> n_entity;
		fin >> n_relation;

		fin >> words;
		fin >> entity;
		fin >> relation;

		logout.record() << "[Dataset] Entity and Relation loaded.";

		logout.record() << "[Data Description] Entity Number = " << entity.size();
		logout.record() << "[Data Description] Relation Number = " << relation.size();
		logout.record() << "[Data Description] Triple Number = " << knowledge.size();
		logout.record() << "[Data Description] Question Number = " << ds_question_test.size();
		logout.record() << "[Data Description] Word Number = " << words.size();

		n_entity = entity.size();
		n_relation = relation.size();
		n_word = words.size();

		logout.record() << "[Data] Done.";
		logout.record();
	}

public:
	tuple<int, int, int> negtive_sample(const tuple<int, int, int>& datum) const
	{
		double prob = relation_hpt[get<1>(datum)] / (relation_hpt[get<1>(datum)] + relation_tph[get<1>(datum)]);
		tuple<int, int, int> neg_sample = datum;

		while (true)
		{
			if (randu() < prob)
			{
				get<2>(neg_sample) = rand() % n_entity;
			}
			else
			{
				get<0>(neg_sample) = rand() % n_entity;
			}

			if (sampling_checker.find(neg_sample) == sampling_checker.end())
				return neg_sample;
		}
	}
};