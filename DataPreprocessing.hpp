#include "Import.hpp"
#include "File.hpp"
#include "Model.hpp"
#include "SemanticModel.hpp"
#include "Task.hpp"
#include "DataProcess.hpp"
#include <boost/format.hpp>  
#include <boost/tokenizer.hpp>  
#include <boost/algorithm/string.hpp>  

map<string, int> Words;
map<int, int> Entity;
map<int, int> Relation;

void deal_question(const string& filein, FormatFile& fout)
{
	vector<pair<tuple<int, int, int>, vector<int>>> ds;

	ifstream fin(filein);
	while (!fin.eof())
	{
		string strline;
		getline(fin, strline);

		stringstream ss(strline);
		int h, r, t;
		ss >> h >> r >> t;

		string ssinput;
		ss >> ssinput;

		vector<int> vin;
		boost::char_separator<char> sep(" \t \"\',.\\?!#%@");
		boost::tokenizer<boost::char_separator<char>>	token(ssinput, sep);
		for (auto i = token.begin(); i != token.end(); ++i)
		{
			if (Words.find(*i) == Words.end())
				Words[*i] = Words.size();

			vin.push_back(Words[*i]);
		}

		ds.push_back(make_pair(make_tuple(h, r, t), vin));
	}

	fout << ds;
}

void deal_knowledge(const string& filein, FormatFile& fout)
{
	map<int, map<int, vector<int> > >   rel_heads;
	map<int, map<int, vector<int> > >   rel_tails;
	vector<double>						relation_tph;
	vector<double>						relation_hpt;
	set<tuple<int, int, int>>			sampling_checker;

	vector<tuple<int, int, int>> ds;
	ifstream fin(filein);
	while (!fin.eof())
	{
		int h, r, t;
		fin >> h >> r >> t;
		if (Entity.find(h) == Entity.end())
			Entity[h] = Entity.size();
		if (Entity.find(t) == Entity.end())
			Entity[t] = Entity.size();
		if (Relation.find(h) == Relation.end())
			Relation[r] = Relation.size();

		ds.push_back(make_tuple(Entity[h], Relation[r], Entity[t]));
	}

	fout << ds;

	for (auto i = ds.begin(); i != ds.end(); ++i)
	{
		rel_heads[get<1>(*i)][get<0>(*i)].push_back(get<2>(*i));
		rel_tails[get<1>(*i)][get<2>(*i)].push_back(get<0>(*i));
	}

	relation_tph.resize(Relation.size());
	for (auto i = 0; i != Relation.size(); ++i)
	{
		double sum = 0;
		double total = 0;
		for (auto ds = rel_heads[i].begin(); ds != rel_heads[i].end(); ++ds)
		{
			++sum;
			total += ds->second.size();
		}
		relation_tph[i] = total / sum;
	}

	relation_hpt.resize(Relation.size());
	for (auto i = 0; i != Relation.size(); ++i)
	{
		double sum = 0;
		double total = 0;
		for (auto ds = rel_tails[i].begin(); ds != rel_tails[i].end(); ++ds)
		{
			++sum;
			total += ds->second.size();
		}
		relation_hpt[i] = total / sum;
	}

	fout << rel_heads;
	fout << rel_tails;
	fout << relation_hpt;
	fout << relation_hpt;
}

void deal_description(const string& filein, FormatFile& fout)
{
	vector<vector<int>> ds(Entity.size() + 1);

	ifstream fin(filein);
	while (!fin.eof())
	{
		string strline;
		getline(fin, strline);

		stringstream ss(strline);
		int ent;
		ss >> ent;

		string ssinput;
		ss >> ssinput;

		vector<int> vin;
		boost::char_separator<char> sep(" \t \"\',.\\?!#%@");
		boost::tokenizer<boost::char_separator<char>>	token(ssinput, sep);
		for (auto i = token.begin(); i != token.end(); ++i)
		{
			if (Words.find(*i) == Words.end())
				continue;

			vin.push_back(Words[*i]);
		}

		if (Entity.find(ent) != Entity.end() && Entity[ent] < ds.size())
			ds[Entity[ent]] = vin;
	}

	fout << ds;
}

void deal_name(const string& filein, FormatFile& fout)
{
	vector<string> ds(Entity.size());

	ifstream fin(filein);
	while (!fin.eof())
	{
		string strline;
		getline(fin, strline);

		stringstream ss(strline);
		int ent;
		ss >> ent;

		string ssinput;
		ss >> ssinput;

		if (Entity.find(ent) != Entity.end() && Entity[ent] < ds.size())
			ds[Entity[ent]] = ssinput;
	}

	fout << ds;
}

int data_prepare()
{
	auto dataset = make_fout("H:\\Data\\Simple Questions.dataset");

	deal_question("H:\\Data\\Simple Questions\\train_uid.txt", dataset);
	deal_question("H:\\Data\\Simple Questions\\valid_uid.txt", dataset);
	deal_question("H:\\Data\\Simple Questions\\test_uid.txt", dataset);
	logout.record() << "[Question] Done.";
	logout.record() << "[Word] #" << Words.size();

	deal_knowledge("H:\\Data\\Simple Questions\\FB2M_uid.txt", dataset);
	logout.record() << "[Knowledge] Done.";
	logout.record() << "[Entity] #" << Entity.size();
	logout.record() << "[Relation] #" << Relation.size();

	deal_description("H:\\Data\\Simple Questions\\FB2M_uid_des.txt", dataset);
	deal_name("H:\\Data\\Simple Questions\\FB2M_uid_name.txt", dataset);

	int n_word = Words.size();
	int n_entity = Entity.size();
	int n_relation = Relation.size();

	dataset << n_word;
	dataset << n_entity;
	dataset << n_relation;

	dataset << Words;
	dataset << Entity;
	dataset << Relation;

	make_close(dataset);

	return 0;
}