import json

def rescale(data, nbx):
    for i in range(len(data["modules"])):
        data["modules"][i]["events"] *= nbx
        data["modules"][i]["time_real"] *= 1e6
        data["modules"][i]["time_thread"] *= 1e6
    data["total"]["events"] *= nbx
    data["total"]["time_real"] *= 1e6
    data["total"]["time_thread"] *= 1e6

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage="%prog in.json out.json")
    parser.add_option("-n", dest="nbx", type=int, default=3564, help="BX per event")
    (options,args) = parser.parse_args()
    fout = args[0].replace(".json","")+".scaled.json"
    data = json.load(open(args[0]))
    rescale(data, options.nbx)
    json.dump(data, open(fout, "w"))
    print(f'Wrote to {fout}')
