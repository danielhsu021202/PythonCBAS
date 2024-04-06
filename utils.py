

class HexUtils:

    def numHexDigits(num: int):
        return len(hex(num)) - 2
    
    def getSeqNumHexNum(num: int, cont, length, LANGUAGE):
        """Constructs the hex string for the sequence number given the number of digits needed to encode the number of contingencies and the maximum sequence length.
        Returns the integer represented by the hex string."""
        nex_num = hex(num)[2:]
        # The hex string for contingency and length, padded with zeros to the appropriate number of digits
        hex_cont = hex(cont)[2:].zfill(HexUtils.numHexDigits(LANGUAGE['NUM_CONTINGENCIES']))
        hex_length = hex(length)[2:].zfill(HexUtils.numHexDigits(LANGUAGE['MAX_SEQUENCE_LENGTH']))
        return int(hex_cont + hex_length + nex_num, 16)
    

class StringUtils:
    
    def parseComplexRange(range_str: str):
        """Parses a complex range string and returns a tuple of the range."""
        try:
            tokens = range_str.split(",")
            values = []
            for token in tokens:
                if "-" in token:
                    values.extend(range(int(token.split("-")[0]), int(token.split("-")[1]) + 1))
                else:
                    values.append(int(token))
            return values
        except:
            return None
        
    def andSeparateList(lst: list, include_verb=False) -> str:
        """Returns a string with the elements of the list separated by 'and'."""
        if len(lst) == 0:
            return ""
        if len(lst) == 1:
            return lst[0] + (" is" if include_verb else "")
        elif len(lst) == 2:
            return lst[0] + " and " + lst[1] + (" are" if include_verb else "")
        else:
            return ", ".join(lst[:-1]) + ", and " + lst[-1] + (" are" if include_verb else "")

    def andSeparateStr(string: str, include_verb=False) -> str:
        """Returns a string with the elements of the string separated by 'and'."""
        return StringUtils.andSeparateList([s.strip() for s in string.split(",")], include_verb)
