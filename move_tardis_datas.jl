

move_from = raw"F:\Public\Tardis_2023\datasets\binance-futures"
move_to = raw"D:\Tardis_2023\datasets\binance-futures"
move_data_categories = ["derivative_ticker", "liquidations"]
move_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]


for symbol in move_symbols
    for category in move_data_categories
        files = readdir(joinpath(move_from, category, symbol))
        for file in files
            date_str = split(file, "_")[1]
            y, m, d = parse.(Int64, split(date_str, "-"))
            if y == 2024 || m > 6 continue end
            from_filename = joinpath(move_from, category, symbol, file)
            target_filename = joinpath(move_to, category, symbol, file)
            if isfile(from_filename) && !isfile(target_filename)
                mkpath(dirname(target_filename))
                cp(from_filename, target_filename)
            end
        end
    end
end













