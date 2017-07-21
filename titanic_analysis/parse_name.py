

def parseName(name):
    if '(' in name:
        return name.split('(')[1].split(' ')[0].replace(')', '')
    else:
        return name.split('.')[1].split(' ')[1].replace(' ','')

def simpleParse(name):
    result = name.split('.')[1].split(' ')[1].replace(' ', '')
    if '(' and ')' in result:
        result.replace('(','')
        return result[1:len(result)-1]
    return result
