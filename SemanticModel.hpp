#pragma once
#include "Import.hpp"
#include "Model.hpp"
#include "Solver.hpp"

class JointMFactorUnit
{
public:
	struct 
	{
		vector<vec> rep_entity;
		vector<vec> rep_relation_subj;
		vector<vec> rep_relation_obj;
		vector<vec> rep_word;
	} _p, _derv_x, _derv_grad;

public:
	AscentAdaDeltaPositive& solver;

public:
	const int		dim;
	const double	sigma;

public:
	JointMFactorUnit(int dim, int entity_count, int relation_count, int word_count, double sigma, AscentAdaDeltaPositive& solver)
		: dim(dim), sigma(sigma), solver(solver)
	{
		_p.rep_entity.resize(entity_count);
		for_each(_p.rep_entity.begin(), _p.rep_entity.end(),
			[=](vec& elem){elem = normalise(randu(dim), 2); });

		_p.rep_relation_subj.resize(relation_count);
		for_each(_p.rep_relation_subj.begin(), _p.rep_relation_subj.end(),
			[=](vec& elem){elem = normalise(randu(dim), 2); });

		_p.rep_relation_obj.resize(relation_count);
		for_each(_p.rep_relation_obj.begin(), _p.rep_relation_obj.end(),
			[=](vec& elem){elem = normalise(randu(dim), 2); });
  
		_p.rep_word.resize(word_count); 
		for_each(_p.rep_word.begin(), _p.rep_word.end(),
			[=](vec& elem){elem = normalise(randu(dim), 2); });

		_derv_x.rep_entity.resize(entity_count);
		for_each(_derv_x.rep_entity.begin(), _derv_x.rep_entity.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_x.rep_relation_subj.resize(relation_count);
		for_each(_derv_x.rep_relation_subj.begin(), _derv_x.rep_relation_subj.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_x.rep_relation_obj.resize(relation_count);
		for_each(_derv_x.rep_relation_obj.begin(), _derv_x.rep_relation_obj.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_x.rep_word.resize(word_count);
		for_each(_derv_x.rep_word.begin(), _derv_x.rep_word.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_grad.rep_entity.resize(entity_count);
		for_each(_derv_grad.rep_entity.begin(), _derv_grad.rep_entity.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_grad.rep_relation_subj.resize(relation_count);
		for_each(_derv_grad.rep_relation_subj.begin(), _derv_grad.rep_relation_subj.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_grad.rep_relation_obj.resize(relation_count);
		for_each(_derv_grad.rep_relation_obj.begin(), _derv_grad.rep_relation_obj.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_grad.rep_word.resize(word_count);
		for_each(_derv_grad.rep_word.begin(), _derv_grad.rep_word.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });
	}

public:
	void save(FormatFile & file)
	{
		file << _p.rep_entity << _p.rep_relation_subj << _p.rep_relation_obj << _p.rep_word;
		file << _derv_x.rep_entity << _derv_x.rep_relation_subj << _derv_x.rep_relation_obj << _derv_x.rep_word;
		file << _derv_grad.rep_entity << _derv_grad.rep_relation_subj << _derv_grad.rep_relation_obj << _derv_grad.rep_word;
	}

	void load(FormatFile & file)
	{
		file >> _p.rep_entity >> _p.rep_relation_subj >> _p.rep_relation_obj >> _p.rep_word;
		file >> _derv_x.rep_entity >> _derv_x.rep_relation_subj >> _derv_x.rep_relation_obj >> _derv_x.rep_word;
		file >> _derv_grad.rep_entity >> _derv_grad.rep_relation_subj >> _derv_grad.rep_relation_obj >> _derv_grad.rep_word;
	}

public:
	double prob(const tuple<int, int, int>& triplet)
	{
		vec& head = _p.rep_entity[get<0>(triplet)];
		vec& tail = _p.rep_entity[get<2>(triplet)];
		vec& relation_head = _p.rep_relation_subj[get<1>(triplet)];
		vec& relation_tail = _p.rep_relation_obj[get<1>(triplet)];

		vec head_feature = head % relation_head;
		vec tail_feature = tail % relation_tail;

		double score = log(sum(head_feature % tail_feature)) * sigma
			- sum(abs(head_feature - tail_feature));

		return score;
	}

	void train_knowledge
		(const tuple<int, int, int>& triplet, const double alpha)
	{
		vec& head = _p.rep_entity[get<0>(triplet)];
		vec& tail = _p.rep_entity[get<2>(triplet)];
		vec& relation_head = _p.rep_relation_subj[get<1>(triplet)];
		vec& relation_tail = _p.rep_relation_obj[get<1>(triplet)];

		vec head_feature = head % relation_head;
		vec tail_feature = tail % relation_tail;
		vec feature = head_feature % tail_feature;
		vec grad = sign(head_feature - tail_feature);

		double sum_feature = sum(feature) + 1e-50;

		vec grad_head =
			-alpha * grad % head_feature / head
			+ alpha * feature / head / sum_feature * sigma;
		vec grad_relation_head =
			-alpha * grad % head_feature / relation_head
			+ alpha * feature / relation_head / sum_feature * sigma;
		vec grad_tail =
			alpha * grad % tail_feature / tail
			+ alpha * feature / tail / sum_feature * sigma;
		vec grad_relation_tail =
			alpha * grad % tail_feature / relation_tail
			+ alpha * feature / relation_tail / sum_feature * sigma;

		solver.gradient(
			_derv_grad.rep_entity[get<0>(triplet)],
			_derv_x.rep_entity[get<0>(triplet)],
			_p.rep_entity[get<0>(triplet)],
			grad_head);

		solver.gradient(
			_derv_grad.rep_entity[get<2>(triplet)],
			_derv_x.rep_entity[get<2>(triplet)],
			_p.rep_entity[get<2>(triplet)],
			grad_tail);

		solver.gradient(
			_derv_grad.rep_relation_subj[get<1>(triplet)],
			_derv_x.rep_relation_subj[get<1>(triplet)],
			_p.rep_relation_subj[get<1>(triplet)],
			grad_relation_head);

		solver.gradient(
			_derv_grad.rep_relation_obj[get<1>(triplet)],
			_derv_x.rep_relation_obj[get<1>(triplet)],
			_p.rep_relation_obj[get<1>(triplet)],
			grad_relation_tail);
	}

	void train_language
		(const int eid, const vector<int>& descriptions, const double alpha)
	{
		vec& entity = _p.rep_entity[eid];
		vec grad_entity = zeros(dim);

		for (auto i = descriptions.begin(); i != descriptions.end(); ++i)
		{
			grad_entity -= alpha * sign(entity - _p.rep_word[*i]);
			vec grad_word = alpha * sign(entity - _p.rep_word[*i]);
			solver.gradient(
				_derv_grad.rep_word[*i],
				_derv_x.rep_word[*i],
				_p.rep_word[*i],
				grad_word);
		}

		solver.gradient(
			_derv_grad.rep_entity[eid],
			_derv_x.rep_entity[eid],
			_p.rep_entity[eid],
			grad_entity);
	}
};

class JointMFactor
	:public Model
{
protected:
	vector<JointMFactorUnit*>	factors;
	vector<vec> rep_entity;
	vector<vec> rep_word;

public:
	AscentAdaDeltaPositive& solver;

public:
	const double	margin;
	const double	alpha;
	const int		dim;
	const int		n_factor;
	const double	sigma;

public:
	JointMFactor(
		const DataSet& ds,
		int n_factor,
		int dim,
		double alpha,
		double margin,
		double sigma,
		AscentAdaDeltaPositive& solver)
		:Model(ds), margin(margin), alpha(alpha), dim(dim), n_factor(n_factor), sigma(sigma), solver(solver)
	{
		logout.record() << "[Name]\tJoint.Multiple.FactorE";
		logout.record() << "[Dimension]\t" << dim;
		logout.record() << "[Learning Rate]\t" << alpha;
		logout.record() << "[Sigma]\t" << sigma;
		logout.record() << "[Training Threshold]\t" << margin;
		logout.record() << "[Factor Number]\t" << n_factor;

		for (auto i = 0; i < n_factor; ++i)
		{
			factors.push_back(new JointMFactorUnit(dim, dm.n_entity + 10, dm.n_relation + 10, dm.n_word + 200, sigma, solver));
		}
	}

public:
	virtual double probability(const tuple<int, int, int>& datum) override
	{
		vec score(n_factor);
		auto i_score = score.begin();
		for (auto factor : factors)
		{
			*i_score++ = factor->prob(datum);
		}

		return sum(score);
	}

	virtual void train_knowledge(const tuple<int, int, int>& datum) override
	{
		tuple<int, int, int> triplet_false = datum;
		if (randu() > 0.5)
		{
			get<0>(triplet_false) = rand() % dm.n_entity;
		}
		else
		{
			get<2>(triplet_false) = rand() % dm.n_entity;
		}

		if (probability(datum) - probability(triplet_false) > margin)
			return;

		for (auto factor : factors)
		{
			factor->train_knowledge(datum, alpha);
			factor->train_knowledge(triplet_false, -alpha);
		}
	}

	virtual void train_language(const int eid, const vector<int>& description) override
	{
		for (auto factor : factors)
		{
			factor->train_language(eid, description, alpha);
		}
	}

	virtual void save(FormatFile & file) override
	{
		for (auto factor : factors)
		{
			factor->save(file);
		}

		file << rep_entity;
		file << rep_word;
	}

	virtual void load(FormatFile & file) override
	{
		for (auto factor : factors)
		{
			factor->load(file);
		}

		file >> rep_entity;
		file >> rep_word;
	}

	virtual vector<int> infer_entity(const vector<int>& des) override
	{
		vector<pair<double, int>> prob_ent(dm.n_entity);
		for (auto i = 0; i < dm.n_entity; ++i)
		{
			prob_ent[i].second = i;
		}

		vec rep_query = zeros(n_factor * dim);
		for (auto ie = 0; ie < dm.n_entity; ++ie)
		{
			prob_ent[ie].first = 1.0;
			for (auto iw = des.begin(); iw != des.end(); ++iw)
			{
				prob_ent[ie].first *= sum(abs(rep_word[*iw] - rep_entity[ie]));
			}
		}

		sort(prob_ent.begin(), prob_ent.end());

		vector<int> res;
		for (int i = 0; i < 10; ++i)
		{
			res.push_back(prob_ent[i].second);
		}

		return res;
	}

	virtual int infer_relation(const vector<int>& des) override
	{
		return 0;
	}

	virtual void train_kernel() override
	{
		Model::train_kernel();

		rep_word.resize(dm.n_word);
		for (auto i = 0; i < dm.n_word; ++i)
		{
			vec tmp;
			for (auto factor : factors)
			{
				tmp = join_cols(tmp, factor->_p.rep_word[i]);
			}
			rep_word[i] = tmp;
		}

		rep_entity.resize(dm.n_entity);
		for (auto i = 0; i < dm.n_entity; ++i)
		{			
			vec tmp;
			for (auto factor : factors)
			{
				tmp = join_cols(tmp, factor->_p.rep_entity[i]);
			}
			rep_entity[i] = tmp;
		}
	}
};
