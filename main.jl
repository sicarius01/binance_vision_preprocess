
include("const.jl")
include("utils.jl")
include("setup.jl")


symbol_list = get_symbol_list(um_path)
@threads for symbol in get_symbol_list(um_path)
    data_type = "aggTrades"
    file_list = get_file_list(um_path, symbol, data_type)
    for (i, file_name) in enumerate(file_list)
        if occursin("\\monthly\\", file_name)
            try
                process_one_month(file_name, symbol, save_dir, interval_sec)
            catch
                println("Error at $(file_name)")
            end
        end
        # if i > 3 break end       
    end
    # break
end









