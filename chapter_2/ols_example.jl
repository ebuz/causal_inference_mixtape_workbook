### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 3319b0cb-c9a9-49c0-b583-df55357d754f
begin
	using Random, Plots, Printf, DataFrames, HTTP, StatFiles, GLM, Distributions, FixedEffectModels, Vcov, LinearAlgebra
	gr()
	Random.seed!(0)
end

# ╔═╡ be69be9a-cafd-11eb-0c5e-87fbf8f251fa
x = randn(10000)

# ╔═╡ dc7d4154-32b3-4741-a25b-416a7e574569
u = randn(10000)

# ╔═╡ db770fef-106e-4107-88c9-faa158486501
y = x * 5.5 + 12 * u

# ╔═╡ 4b3680e2-9754-4343-9391-30f1369bb958
function mean(arr)
	return sum(arr)/length(arr)
end

# ╔═╡ 93d0a964-704d-47c3-a9b3-def9a6d64a92
function variance(arr)
	m_arr = mean(arr)
	return mean((arr .- m_arr).^2)
end

# ╔═╡ b1040753-b0cd-4258-92e3-ff87f33b00ca
function covariance(arr1, arr2)
	return mean(arr1 .* arr2) - mean(arr1)*mean(arr2)
end

# ╔═╡ 63e06157-77fb-4536-aa7e-89b74dba414f
function estimate_b1(x, y)
	return covariance(x, y)/variance(x)
end

# ╔═╡ f69f49c5-c1b3-4574-a3e3-e43f884b6d2e
function estimate_b0(x, y, b1)
	return mean(y) - b1 * mean(x)
end

# ╔═╡ 4cd8c723-1de1-43ef-adbf-71893271208c
b1 = estimate_b1(x, y)

# ╔═╡ 6fd4dc62-37d2-4622-952e-c3007018bda1
b0 = estimate_b0(x, y, b1)

# ╔═╡ 7f764c0e-60cc-4f00-b81a-194261a76422
function ols(x, y)
	b1 = estimate_b1(x, y)
	b0 = estimate_b0(x, y, b1)
	f = function lm_predict(x1)
		return b0 .+ (b1 .* x1)
	end
	return f
end

# ╔═╡ b49f3a3d-dde2-4b9b-86b4-009cb137cb2d
yhat = b0 .+ (b1 .* x)

# ╔═╡ 8e223297-bc00-45d0-9bee-bfc2b8d00dc6
residuals = y .- yhat

# ╔═╡ 5cbc4ac0-1bed-4499-b276-fa832bdc8987
begin
	scatter(x, y, title = "OLS Regression Line", xlabel = "x", ylabel = "y",
		label = "data"
	)
	plot!(x, yhat, lw = 3, label = "regression"
	)
	annotate!([
			(-1.5, 35, Plots.text(@sprintf("Intercept = %.2f", b0), :orange)),
			(1.5, -35, Plots.text(@sprintf("Slope = %.2f", b1), :green))
			])
	plot!([-1.75, 0], [30, b0], lw = 2, label = "", color = :orange, arrow = true)
	plot!([1.75, 1], [-30, 1], lw = 2, label = "", color = :green, arrow = true)

end

# ╔═╡ 5b80100c-b508-46d3-a844-5bd0c822a1ee
scatter(yhat, residuals, xlabel = "Fitted values", ylabel = "Residuals",
	label=""
	)

# ╔═╡ d94626b5-53f7-43a3-ad61-39ca3e6be77c
begin
	tb = DataFrame(
		x = 9 * randn(10),
		u = 36 * randn(10))
	tb[!, "y"] = 3 * tb.x + 2 * tb.u
	pred = ols(tb.x, tb.y)
	tb[!, "yhat"] = pred(tb.x)
	tb[!, "uhat"] = tb[!, "y"] - tb[!, "yhat"]
	tb
end

# ╔═╡ 94d1b9fd-baf7-4166-a3d4-8291a7442edd
mapcols(sum, tb)
# alternatively: map(sum, eachcol(df))

# ╔═╡ 70c31e20-4e99-4084-b764-4374601fcfa2
1 - sum(tb.uhat .^ 2) / sum((tb.y .- mean(tb.y)) .^ 2)

# ╔═╡ 88a38f00-f884-48e3-8f84-0728fab10f12
begin
	b1s = Array{Float64}(undef, 1000)
	for i in 1:1000
		tb = DataFrame(
			x = 9 * randn(10000),
			u = 36 * randn(10000))
		tb[!, "y"] = 3 .+ ((2 .* tb.x) + tb.u)
		b1 = estimate_b1(tb.x, tb.y)
		b1s[i] = b1
	end
end

# ╔═╡ dc9f733d-c788-4d2c-a852-c1eaac6c7a2c
histogram(b1s)

# ╔═╡ 0ca66411-8505-47da-88e3-5d1e44678af2
DataFrame(mean = mean(b1s), variance = sqrt(variance(b1s)))

# ╔═╡ 76d45c4e-7b10-4492-b1ff-f6caa06e90aa
download("https://raw.github.com/scunning1975/mixtape/master/auto.dta", "../data/auto.dta")

# ╔═╡ ea1c3ca8-4ebd-45f8-9020-95ac9f8cf183
begin
	auto = DataFrame(load("../data/auto.dta"))
	auto.length = auto.length .- mean(auto.length)
end

# ╔═╡ ea95aaa2-5156-4f77-8d6e-abb111a3d9ba
auto

# ╔═╡ 32171b78-bee7-4242-b609-7522585bd627
begin
	lm1 = lm(@formula(price ~ length), auto)
	lm2 = lm(@formula(price ~ length + weight + headroom + mpg), auto)
	lm_aux = lm(@formula(length ~ weight + headroom + mpg), auto)
	auto.length_resid = GLM.residuals(lm_aux)
	lm2_alt = lm(@formula(price ~ length_resid), auto)
end

# ╔═╡ 8d85f40a-9904-4055-a297-2279db7e257f
y_single = DataFrame(
	price = coef(lm2_alt)[1] .+ coef(lm1)[2] * auto.length_resid,
	length_resid = auto.length_resid
	)

# ╔═╡ 658b8e19-8c0b-4a16-8960-a87be61c6d45
y_multi = DataFrame(
	price = coef(lm2_alt)[1] .+ coef(lm2_alt)[2] * auto.length_resid,
	length_resid = auto.length_resid
	)

# ╔═╡ 707ee46f-f7ee-4679-9644-b175e526d4da
lm1

# ╔═╡ b444ec65-a576-4e0d-a70d-60c77334db70
lm2

# ╔═╡ 2e993bea-1e8b-4006-b829-48926a21a1f1
begin
	scatter(auto.length_resid, auto.price, label = "")
	plot!(y_multi.length_resid, y_multi.price, label = "Multi", color = :blue)
	plot!(y_single.length_resid, y_single.price, label = "Single", color = :red)
end

# ╔═╡ 845a5d78-b063-49a2-acd6-cdc72ce11c38
# biased due to degrees of freedom adjustment
sqrt(mean(GLM.residuals(lm1).^2)/sum((auto.length .- mean(auto.length)).^2))

# ╔═╡ 5e52a943-3acc-41e1-89c7-d2347d8edb2f
# adjusted, note how this matches the lm1 model summary of the coefficient of length
# reproduced below
sqrt((sum(GLM.residuals(lm1).^2)/(nrow(auto)-2))/sum((auto.length .- mean(auto.length)).^2))

# ╔═╡ 2c53de1a-147d-4f60-b32b-c6c7b8f82809
stderror(lm1)[2]

# ╔═╡ 5ecca4c0-8b0e-44c7-9fe8-2178c0f070b8
# adapted from python code so might not be most efficient or idiomatic
function gen_cluster(;
		param::Array{<:Real} = [.1, .5], n::Integer = 1000,
		n_cluster::Integer = 50, rho::Real = 0.5)
	Sigma_i = reshape([1, 0, 0, 1 - rho], 2, 2)
	dist_i = Distributions.MvNormal(zeros(2), Sigma_i)
	values_i = rand(dist_i, n)
	cluster_name = repeat(collect(1:n_cluster), inner = n ÷ n_cluster)
	Sigma_cl = reshape([1, 0, 0, rho], 2, 2)
	dist_cl = Distributions.MvNormal(zeros(2), Sigma_cl)
	values_cl = rand(dist_cl, n_cluster)
	
	x = values_i[1, :] + repeat(values_cl[1, :], inner = n ÷ n_cluster)
	error = values_i[2, :] + repeat(values_cl[2, :], inner = n ÷ n_cluster)
	y = param[1] .+ ((param[2] * x) + error)
	return DataFrame(x = x, y = y, cluster = cluster_name)
end

# ╔═╡ c1c72228-87d1-4c11-887f-1d87157818f2
function cluster_sim(;
		param::Array{<:Real} = [.1, .5], n::Integer = 1000, n_cluster::Integer = 50, rho::Real = 0.5,
		cluster_robust::Bool = false)
	df = gen_cluster(param = param, n = n, n_cluster = n_cluster, rho = rho)
	if cluster_robust
		fit = FixedEffectModels.reg(df, @formula(y ~ x), Vcov.cluster(:cluster))
	else
		fit = GLM.lm(@formula(y ~ x), df)
	end
	b1 = coef(fit)[2]
	Sigma = vcov(fit)
	se = sqrt(diag(Sigma)[2])
	ci95 = se * 1.96
	b1_ci95 = (b1 - ci95, b1 + ci95)
	b1_se = stderror(fit)[2]
	return (b1, se, b1_ci95...)
end

# ╔═╡ 70c309b3-419c-4ac7-84a8-3bd12d3d6def
function run_cluster_sim(;
		n_sims = 1000,
		param = [.1, .5], n = 1000, n_cluster = 50, rho = 0.5,
		cluster_robust = false)
	res = [cluster_sim(param = param, n = n, rho = rho, cluster_robust = cluster_robust) for x in 1:n_sims]
	df = DataFrame(res)
	rename!(df, ["b1", "se_b1", "ci95_lower", "ci95_upper"])
	df.param_caught = (df.ci95_lower .<= param[2]) .& (df.ci95_upper .>= param[2])
	return df
end

# ╔═╡ f6ceb576-479b-4539-b18d-bae55895ebbb
begin
	sim_params = [.4, 0]
	sim_nocluster = run_cluster_sim(n_sims = 1000, param = sim_params,
		rho = 1e-32,
		cluster_robust = false)
	sort!(sim_nocluster, [:b1])
	sim_nocluster.id = collect(1:nrow(sim_nocluster))
end

# ╔═╡ d4ab8416-c581-42f8-9f79-34499df9abba
begin
	histogram(sim_nocluster.b1, label="")
	vline!([sim_params[2]], color = :green, lw = 3, label="True value")
end

# ╔═╡ 7a192c12-df9d-464b-a8ba-2529bf2211a9
let # we use a let here to avoid filling the global scope with variables we want to reuse
	sim_nocluster.color = map(x -> ifelse(x, "darkgray", "lightgray"), sim_nocluster.param_caught)
	sim_nocluster.linetype = map(x -> ifelse(x, :solid, :dot), sim_nocluster.param_caught)

	sim_subset = sim_nocluster[1:10:nrow(sim_nocluster), :]
	sim_subset_caught = sim_subset[sim_subset.param_caught, :]
	sim_subset_not_caught = sim_subset[.!sim_subset.param_caught, :]


	scatter(sim_subset_caught.b1, sim_subset_caught.id,
		xerror=1.96 .* sim_subset_caught.se_b1, label="Hit",
		ylabel="Simulation ID",
		xlabel="b1",
		markerstrokecolor=sim_subset_caught.color,
		color=sim_subset_caught.color,
		markercolor=sim_subset_caught.color,
		markerstrokestyle=sim_subset_caught.linetype,
		linestyle=sim_subset_caught.linetype, flip=true
	)
	scatter!(sim_subset_not_caught.b1, sim_subset_not_caught.id,
		xerror=1.96 .* sim_subset_not_caught.se_b1, label="Miss",
		ylabel="Simulation ID",
		xlabel="b1",
		markerstrokecolor=sim_subset_not_caught.color,
		color=sim_subset_not_caught.color,
		markercolor=sim_subset_not_caught.color,
		markerstrokestyle=sim_subset_not_caught.linetype,
		linestyle=sim_subset_not_caught.linetype, flip=true
	)
	vline!([sim_params[2]], linestyle=:dash, label="True population parameter",
		legend=:bottomright
	)
end

# ╔═╡ d31c40fb-b449-48fc-bdf2-e5a76afaa3a6
begin
	sim_cluster = run_cluster_sim(n_sims = 1000, param = sim_params, cluster_robust = false)
	histogram(sim_cluster.b1, label="")
	vline!([sim_params[2]], color = :green, lw = 3, label="True value")
	sort!(sim_cluster, [:b1])
	sim_cluster.id = collect(1:nrow(sim_cluster))

end

# ╔═╡ 0a61da6e-a299-423c-978e-bf1d8ac01bb7
1 - mean(sim_cluster.param_caught)

# ╔═╡ 1f5d7592-8c10-4b8c-bb4e-7cdf08361dbc
let
	sim_cluster.color = map(x -> ifelse(x, "darkgray", "lightgray"), sim_cluster.param_caught)
	sim_cluster.linetype = map(x -> ifelse(x, :solid, :dot), sim_cluster.param_caught)

	sim_subset = sim_cluster[1:10:nrow(sim_cluster), :]
	sim_subset_caught = sim_subset[sim_subset.param_caught, :]
	sim_subset_not_caught = sim_subset[.!sim_subset.param_caught, :]


	scatter(sim_subset_caught.b1, sim_subset_caught.id,
		xerror=1.96 .* sim_subset_caught.se_b1, label="Hit",
		ylabel="Simulation ID",
		xlabel="b1",
		markerstrokecolor=sim_subset_caught.color,
		color=sim_subset_caught.color,
		markercolor=sim_subset_caught.color,
		markerstrokestyle=sim_subset_caught.linetype,
		linestyle=sim_subset_caught.linetype, flip=true
	)
	scatter!(sim_subset_not_caught.b1, sim_subset_not_caught.id,
		xerror=1.96 .* sim_subset_not_caught.se_b1, label="Miss",
		ylabel="Simulation ID",
		xlabel="b1",
		markerstrokecolor=sim_subset_not_caught.color,
		color=sim_subset_not_caught.color,
		markercolor=sim_subset_not_caught.color,
		markerstrokestyle=sim_subset_not_caught.linetype,
		linestyle=sim_subset_not_caught.linetype, flip=true
	)
	vline!([sim_params[2]], linestyle=:dash, label="True population parameter",
		legend=:bottomright
	)
end

# ╔═╡ 046c1ea7-6a0c-4c0b-8108-6a829d0587a4
begin
	sim_cluster_robust = run_cluster_sim(n_sims = 1000, param = sim_params, cluster_robust = true)
	histogram(sim_cluster_robust.b1, label="")
	vline!([sim_params[2]], color = :green, lw = 3, label="True value")
	sort!(sim_cluster_robust, [:b1])
	sim_cluster_robust.id = collect(1:nrow(sim_cluster_robust))

end

# ╔═╡ f7a31a95-3ae4-4ab8-a944-73e087bbb9fc
1 - mean(sim_cluster_robust.param_caught)

# ╔═╡ 8d859b5a-70e8-490e-ba71-81fb44d4b0ac
let
	sim_cluster_robust.color = map(x -> ifelse(x, "darkgray", "lightgray"), sim_cluster_robust.param_caught)
	sim_cluster_robust.linetype = map(x -> ifelse(x, :solid, :dot), sim_cluster_robust.param_caught)

	sim_subset = sim_cluster_robust[1:10:nrow(sim_cluster_robust), :]
	sim_subset_caught = sim_subset[sim_subset.param_caught, :]
	sim_subset_not_caught = sim_subset[.!sim_subset.param_caught, :]


	scatter(sim_subset_caught.b1, sim_subset_caught.id,
		xerror=1.96 .* sim_subset_caught.se_b1, label="Hit",
		ylabel="Simulation ID",
		xlabel="b1",
		markerstrokecolor=sim_subset_caught.color,
		color=sim_subset_caught.color,
		markercolor=sim_subset_caught.color,
		markerstrokestyle=sim_subset_caught.linetype,
		linestyle=sim_subset_caught.linetype, flip=true
	)
	scatter!(sim_subset_not_caught.b1, sim_subset_not_caught.id,
		xerror=1.96 .* sim_subset_not_caught.se_b1, label="Miss",
		ylabel="Simulation ID",
		xlabel="b1",
		markerstrokecolor=sim_subset_not_caught.color,
		color=sim_subset_not_caught.color,
		markercolor=sim_subset_not_caught.color,
		markerstrokestyle=sim_subset_not_caught.linetype,
		linestyle=sim_subset_not_caught.linetype, flip=true
	)
	vline!([sim_params[2]], linestyle=:dash, label="True population parameter",
		legend=:bottomright
	)
end

# ╔═╡ c78824ad-3073-42cf-ab52-08c77ecaa1e7


# ╔═╡ Cell order:
# ╠═3319b0cb-c9a9-49c0-b583-df55357d754f
# ╠═be69be9a-cafd-11eb-0c5e-87fbf8f251fa
# ╠═dc7d4154-32b3-4741-a25b-416a7e574569
# ╠═db770fef-106e-4107-88c9-faa158486501
# ╠═4b3680e2-9754-4343-9391-30f1369bb958
# ╠═93d0a964-704d-47c3-a9b3-def9a6d64a92
# ╠═b1040753-b0cd-4258-92e3-ff87f33b00ca
# ╠═63e06157-77fb-4536-aa7e-89b74dba414f
# ╠═f69f49c5-c1b3-4574-a3e3-e43f884b6d2e
# ╠═4cd8c723-1de1-43ef-adbf-71893271208c
# ╠═6fd4dc62-37d2-4622-952e-c3007018bda1
# ╠═7f764c0e-60cc-4f00-b81a-194261a76422
# ╠═b49f3a3d-dde2-4b9b-86b4-009cb137cb2d
# ╠═8e223297-bc00-45d0-9bee-bfc2b8d00dc6
# ╠═5cbc4ac0-1bed-4499-b276-fa832bdc8987
# ╠═5b80100c-b508-46d3-a844-5bd0c822a1ee
# ╠═d94626b5-53f7-43a3-ad61-39ca3e6be77c
# ╠═94d1b9fd-baf7-4166-a3d4-8291a7442edd
# ╠═70c31e20-4e99-4084-b764-4374601fcfa2
# ╠═88a38f00-f884-48e3-8f84-0728fab10f12
# ╠═dc9f733d-c788-4d2c-a852-c1eaac6c7a2c
# ╠═0ca66411-8505-47da-88e3-5d1e44678af2
# ╠═76d45c4e-7b10-4492-b1ff-f6caa06e90aa
# ╠═ea1c3ca8-4ebd-45f8-9020-95ac9f8cf183
# ╠═ea95aaa2-5156-4f77-8d6e-abb111a3d9ba
# ╠═32171b78-bee7-4242-b609-7522585bd627
# ╠═8d85f40a-9904-4055-a297-2279db7e257f
# ╠═658b8e19-8c0b-4a16-8960-a87be61c6d45
# ╠═707ee46f-f7ee-4679-9644-b175e526d4da
# ╠═b444ec65-a576-4e0d-a70d-60c77334db70
# ╠═2e993bea-1e8b-4006-b829-48926a21a1f1
# ╠═845a5d78-b063-49a2-acd6-cdc72ce11c38
# ╠═5e52a943-3acc-41e1-89c7-d2347d8edb2f
# ╠═2c53de1a-147d-4f60-b32b-c6c7b8f82809
# ╠═5ecca4c0-8b0e-44c7-9fe8-2178c0f070b8
# ╠═c1c72228-87d1-4c11-887f-1d87157818f2
# ╠═70c309b3-419c-4ac7-84a8-3bd12d3d6def
# ╠═f6ceb576-479b-4539-b18d-bae55895ebbb
# ╠═d4ab8416-c581-42f8-9f79-34499df9abba
# ╠═7a192c12-df9d-464b-a8ba-2529bf2211a9
# ╠═d31c40fb-b449-48fc-bdf2-e5a76afaa3a6
# ╠═0a61da6e-a299-423c-978e-bf1d8ac01bb7
# ╠═1f5d7592-8c10-4b8c-bb4e-7cdf08361dbc
# ╠═046c1ea7-6a0c-4c0b-8108-6a829d0587a4
# ╠═f7a31a95-3ae4-4ab8-a944-73e087bbb9fc
# ╠═8d859b5a-70e8-490e-ba71-81fb44d4b0ac
# ╠═c78824ad-3073-42cf-ab52-08c77ecaa1e7
