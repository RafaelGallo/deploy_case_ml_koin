-- Total de registros
SELECT COUNT(*) FROM dados_tratados;

-- Distribuição da variável alvo
SELECT over30_mob3, COUNT(*) 
FROM dados_tratados
GROUP BY over30_mob3;

--Inadimplência por UF
SELECT uf, over30_mob3, COUNT(*) 
FROM dados_tratados
GROUP BY uf, over30_mob3;

--Média de valor de compra por classe
SELECT over30_mob3, AVG(valor_compra)
FROM dados_tratados
GROUP BY over30_mob3;

--Top 10 maiores compras
SELECT *
FROM dados_tratados
ORDER BY valor_compra DESC
LIMIT 10;