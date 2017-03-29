#pragma once
#include "Import.hpp"
#include "File.hpp"
#include "Model.hpp"
#include "SemanticModel.hpp"
#include "Task.hpp"
#include <boost/format.hpp>  
#include <boost/tokenizer.hpp>  
#include <boost/algorithm/string.hpp>  

template<typename S>
class Namer
{
public:
	map<S, int> finder;
	bool locked;

public:
	Namer()
		:locked(false)
	{
		;
	}

	int get(const S& a)
	{
		auto i = finder.find(a);
		if (i == finder.end() && !locked)
		{
			finder[a] = finder.size();
			return finder.size() - 1;
		}
		else if (i == finder.end() && locked)
		{
			return finder.size();
		}
		else
		{
			return i->second;
		}
	}

	void lock()
	{
		locked = true;
	}
};

Namer<string> uid_ent_namer;
Namer<string> uid_rel_namer;
Namer<string> word_namer;

//void renamer(const string& file_path, const string& name)
//{
//	vector<pair<tuple<int, int, int>, vector<int>>> ds_question;
//
//	boost::char_separator<char> sep(" \t \"\',.\\?!#%@");
//	ifstream fin_des(file_path);
//	while (!fin_des.eof())
//	{
//		int id_h, id_r, id_t;
//		string strin;
//		getline(fin_des, strin);
//		stringstream ss(strin);
//		ss >> id_h >> id_r >> id_t;
//
//		boost::tokenizer<boost::char_separator<char>> token(strin, sep);
//
//		vector<string> holder;
//		vector<int> entity_description;
//		for (auto i = token.begin(); i != token.end(); ++i)
//		{
//			holder.push_back(*i);
//		}
//
//		if (holder.size() < 3)
//			continue;
//
//		id_h = uid_ent_namer.get(id_h);
//		id_r = uid_rel_namer.get(id_r);
//		id_t = uid_ent_namer.get(id_t);
//
//		for (auto i = 3; i < holder.size(); ++i)
//		{
//			entity_description.push_back(word_namer.get(holder[i]));
//		}
//
//		ds_question.push_back(make_pair(make_tuple(id_h, id_r, id_t), entity_description));
//	}
//	fin_des.close();
//	make_fout("H:\\Data\\Simple Questions\\" + name) << ds_question;
//}

void reknowledge(const string& filename, const string& name)
{
	vector<tuple<int, int, int>> knowledge;

	ifstream fin(filename);
	while (!fin.eof())
	{
		string h, r, t;
		fin >> h >> r >> t;

		int id_h, id_r, id_t;
		id_h = uid_ent_namer.get(h);
		id_r = uid_rel_namer.get(r);
		id_t = uid_ent_namer.get(t);

		if (id_h < uid_ent_namer.finder.size() && id_t < uid_ent_namer.finder.size())
			knowledge.push_back(make_tuple(id_h, id_r, id_t));
	}
	fin.close();
	make_fout("H:\\Data\\Knowledge Embedding\\FB15K\\" + name) << knowledge;
	logout.record() << "Done!";
}

void data_preprocess()
{
	reknowledge("H:\\Data\\Knowledge Embedding\\FB15K\\train.txt", "train.bin");
	reknowledge("H:\\Data\\Knowledge Embedding\\FB15K\\test.txt", "test.bin");
	reknowledge("H:\\Data\\Knowledge Embedding\\FB15K\\dev.txt", "dev.bin");

	vector<vector<int>> description;
	description.resize(uid_ent_namer.finder.size());
	boost::char_separator<char> sep(" \t \"\',.\\?!#%@-;:=+-()[]");
	ifstream fin_des("H:\\Data\\Knowledge Embedding\\FB15K\\description.txt");
	while (!fin_des.eof())
	{
		string strin;
		getline(fin_des, strin);
		transform(strin.begin(), strin.end(), strin.begin(), ::tolower);

		boost::tokenizer<boost::char_separator<char>> token(strin, sep);
		string uid;
		vector<int> entity_description;
		for (auto i = token.begin(); i != token.end(); ++i)
		{
			if (i == token.begin())
			{
				uid = *i;
			}
			else
			{
				entity_description.push_back(word_namer.get(*i));
			}
		}

		if (entity_description.size() != 0)
			description[uid_ent_namer.get(uid)] = entity_description;
	}

	fin_des.close();
	make_fout("H:\\Data\\Knowledge Embedding\\FB15K\\description.bin") << description;
	logout.record() << "Done!";

	make_fout("H:\\Data\\Knowledge Embedding\\FB15K\\words.bin") << word_namer.finder;
	logout.record() << "Done!";
}