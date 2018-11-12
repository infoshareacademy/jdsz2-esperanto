-- ESP-12
-- dla agenta nr 168 dokonano 47 tys analiz, z czego 72% zostala wyplacona


with wniosek_analiza as
  (
  Select w.id, w.stan_wniosku, w.data_utworzenia, a.id_wniosku, a.status, a.data_utworzenia, a.id_agenta,
      count(case when w.stan_wniosku = 'wyplacony' then 1 end) over (partition by a.id_agenta)
      /
      count(w.id) over (partition by a.id_agenta)::numeric as proc_wypl_agent
  from wnioski w
  join analizy_wnioskow a on a.id_wniosku = w.id
  where a.id_agenta is not null
  group by w.id, w.data_utworzenia, a.id_wniosku, a.status, a.data_utworzenia, a.id_agenta
  order by 8 desc
  )
Select id_agenta, count(id_agenta), proc_wypl_agent
from wniosek_analiza
group by id_agenta, proc_wypl_agent
having count(id_agenta) > 1
order by 2 desc


