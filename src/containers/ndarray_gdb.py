###############################################################################
# Copy this file to "~/.gdb" (creating the directory, if not
# present) and then apppend the following line to the file "~/.gdbinit"
# (again, creating it if, not already present):
###############################################################################
# set print pretty
# source ~/.gdb/ndarray_gdb.py
###############################################################################

import gdb

simple_types = {
    'bool',
    'char', 'unsigned char',
    'short', 'unsigned short',
    'int', 'unsigned int',
    'long', 'unsigned long',
    'long long', 'unsigned long long',
    'float', 'double'
}

class NDArrayDynamicArrayPrinter:
    def __init__(self, instance):
        self.instance = instance
        itype = self.instance.type.strip_typedefs()
        self.size = int(str(self.instance['m_size']))
        self.packet_count = int(str(self.instance['m_packets_allocated']))
        self.packet_type = itype.template_argument(0)
        self.packet_size = self.packet_type.sizeof
        self.data = int(str(instance['m_packets']['_M_t']['_M_t']['_M_head_impl']), 0)
        self.limit = 20

    def to_string(self):
        values = []
        for i in range(self.packet_count):
            addr = int(self.data) + self.packet_size * i
            cmd = '*((%s *) 0x%x)' % (str(self.packet_type), addr)
            value = str(gdb.parse_and_eval(cmd))
            assert value[-1] == ']'
            values += value[value.rfind('[')+1:-1].split(', ')
            if len(values) > self.size:
                values = values[0:self.size]
                break
            if len(values) > self.limit:
                break
        if len(values) > self.limit:
            values = values[0:self.limit]
            values.append(".. %i skipped .." % (self.size - self.limit))
        return '[' + ', '.join(values) + ']'

regexp = r'(nd::)?(array|view)'

p = gdb.printing.RegexpCollectionPrettyPrinter("nd::array")
p.add_printer("static", regexp_combined, NDArrayStaticArrayPrinter)
p.add_printer("dynamic", r'^(enoki::)?DynamicArray(Impl)?<.+>$', NDArrayDynamicArrayPrinter)

o = gdb.current_objfile()
gdb.printing.register_pretty_printer(o, p)
